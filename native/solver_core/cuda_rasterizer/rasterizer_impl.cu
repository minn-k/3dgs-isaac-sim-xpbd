/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid,
	int2* rects)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		if(rects == nullptr)
			getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);
		else
			getRect(points_xy[idx], rects[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;// 타일 번호 (예: 5번 타일)
				key <<= 32;// 왼쪽으로 32칸 밀기! [ 5 | 00000000 ]
				key |= *((uint32_t*)&depths[idx]);// 하위 32비트에 깊이(Z값) 끼워넣기!
				gaussian_keys_unsorted[off] = key;// 최종 Key: [ 타일 번호 | 깊이 값 ]
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
		if (idx == L - 1)
			ranges[currtile].y = L;
	}
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	FORWARD::ChainMail& cm, 
	std::vector<int>& activeSet,
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	float* means3D,
	int nbr_K,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,


	const float _rotatingModifier_COV3D_Matrix_x,
	const float _rotatingModifier_COV3D_Matrix_y,
	const float _rotatingModifier_COV3D_Matrix_z,
	const float _rotatingModifier_COV2D_Matrix_x,
	const float _rotatingModifier_COV2D_Matrix_y,
	const float _rotatingModifier_COV2D_Matrix_z,


	const float _pivotRotX,
	const float _pivotRotY,
	const float _pivotRotZ,

	float t,
	bool _wave,
	bool _twist,
	bool _bubble,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	bool antialiasing,
	int* id_buffer,
	int* radii,
	int* rects,
	float* boxmin,
	float* boxmax)
{
	// ==========================================
// [1] 함수 시작부: 정적 변수 및 이벤트 초기화
// ==========================================
	static cudaEvent_t start_total, end_total;
	static cudaEvent_t start_render, end_render;
	static std::vector<float> total_log, render_log;
	static int frame_count = 0;
	static bool timer_init = false;

	if (!timer_init) {
		cudaEventCreate(&start_total); cudaEventCreate(&end_total);
		cudaEventCreate(&start_render); cudaEventCreate(&end_render);
		timer_init = true;
	}

	// 전체 프레임 측정 시작
	cudaEventRecord(start_total);
	//원근 투영을 위한 카메라의 초점거리 계산
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	//ImageState 초기화 (렌더링 결과용)
	int img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	float3 minn = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
	float3 maxx = { FLT_MAX, FLT_MAX, FLT_MAX };
	if (boxmin != nullptr)
	{
		minn = *((float3*)boxmin);
		maxx = *((float3*)boxmax);
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	//Gaussians를 3D에서 2D 이미지로 투영하기 위한 모든 연산
	// output = geomState
	//이미지 상의 2D 위치, 깊이값
	//2D 공분산 행렬
	//RGB 색상
	//투명도
	//영향을 받는 타일(tile) 정보 등
	FORWARD::preprocess(
		cm, activeSet,
		P, D, M,
		means3D,
		nbr_K,
		(glm::vec3*)scales,
		scale_modifier,


		_rotatingModifier_COV3D_Matrix_x,
		_rotatingModifier_COV3D_Matrix_y,
		_rotatingModifier_COV3D_Matrix_z,
		_rotatingModifier_COV2D_Matrix_x,
		_rotatingModifier_COV2D_Matrix_y,
		_rotatingModifier_COV2D_Matrix_z,


		_pivotRotX,
		_pivotRotY,
		_pivotRotZ,


		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,//각 가우시안이 걸쳐있는 타일 수
		prefiltered,
		(int2*)rects,
		minn,
		maxx,
		antialiasing,
		t,
		_wave,
		_twist,
		_bubble
	);
	cudaEventRecord(start_render);
	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size,
		geomState.tiles_touched, geomState.point_offsets, P);

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost);

	if (num_rendered == 0)
		return 0;

	int binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (//가우시안이 걸친 타일 개수만큼 가우시안의 고유 ID(Index)를 복제하면서, 정렬을 위한 64비트 키(32타일번호|32깊이값)를 만듬
		//복제 1: Key = [ 1 | 0.5 ], Value = 7
		//복제 2: Key = [2 | 0.5], Value = 7
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid,
		(int2*)rects
		);

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	//GPU 정렬. 메모리 안에는 타일 번호별로 묶임
	//각 타일에는 (Front-to-back)대로 정렬된 가우시안 리스트(point_list)생성.
	cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit);

	cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2));

	// Identify start and end of per-tile workloads in sorted list
	identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (//정렬된 리스트를 쭉 훑으면서 타일 번호가 바뀌는 경계선을 찾음
		num_rendered,
		binningState.point_list_keys,
		imgState.ranges//ranges[5] = {100, 150} 이라면, 5번 타일에 그려져야 할 가우시안들은 정렬된 리스트의 100번째부터 149번째에 모여 있다는 뜻
		);

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	if (id_buffer != nullptr)//id_buffer가 있다면 -1로 초기화
	{
		const size_t id_bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * sizeof(int);
		cudaMemset(id_buffer, -1, id_bytes);
	}
	FORWARD::render(
		tile_grid, 
		block,//1개의 CUDA 블록 = 1개의 타일(16X16 픽셀) = 256명의 스레드
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color,
		id_buffer);
	// 렌더링 측정 종료
	cudaEventRecord(end_render);
	// ==========================================
// [2] 함수 종료부: 전체 프레임 측정 완료 및 출력
// ==========================================
	cudaEventRecord(end_total);
	cudaEventSynchronize(end_total); // 전체 파이프라인 완료 대기

	// ---------------------------------------------------------
	// [4] 결과 계산 및 로그 출력
	// ---------------------------------------------------------
	float frame_ms = 0.0f, render_ms = 0.0f;
	cudaEventElapsedTime(&frame_ms, start_total, end_total);
	cudaEventElapsedTime(&render_ms, start_render, end_render);

	total_log.push_back(frame_ms);
	render_log.push_back(render_ms);
	frame_count++;

	if (frame_count >= 100) {
		float avg_total = std::accumulate(total_log.begin(), total_log.end(), 0.0f) / 100.0f;
		float avg_render = std::accumulate(render_log.begin(), render_log.end(), 0.0f) / 100.0f;

		printf("\n==============================================\n");
		printf(" [ TABLE 2 DATA COLLECTION - AVG 100 FRAMES ] \n");
		printf("----------------------------------------------\n");
		printf(" * Rendering (Sorting+Raster) : %.4f ms\n", avg_render); // 표의 Render 칸
		printf(" * Total Algorithm Latency    : %.4f ms\n", avg_total);  // 표의 Total 칸
		printf(" * System Real-time FPS       : %.2f FPS\n", 1000.0f / avg_total);
		printf(" * Processed Gaussians        : %d\n", P);
		printf("==============================================\n\n");

		total_log.clear(); render_log.clear(); frame_count = 0;
	}
	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	bool antialiasing)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor);

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		opacities,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		dL_dopacity,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot,
		antialiasing);
}
