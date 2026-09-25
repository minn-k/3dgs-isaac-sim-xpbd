// Isaac Sim(Python ctypes)에서 APG-GS XPBD 물리를 부르기 위한 C API.
// 물리 코드는 SIBR 의 CudaRasterizer.lib (forward.cu) 를 그대로 링크한다 — 여기에는 입력 변환과 얇은 래퍼만 있다.
//
// 입력 순서는 Isaac(USD) 순서: prepare_splat.py 의 크롭 PLY 순서, import_graph.py 가 그 순서로 바꾼 그래프.
// 좌표는 원본 3DGS 좌표(Isaac 월드 변환 전)이다. 바닥·중력도 같은 좌표계·단위로 넘긴다.
// 상태는 전역 하나 (forward.cu 의 물리 상태가 전역이므로 인스턴스도 하나뿐이다).
#include <cstdio>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "forward.h"

#define XAPI extern "C" __declspec(dllexport)

namespace {
FORWARD::ChainMail* g_cm = nullptr;
glm::vec3* d_scales = nullptr;   // 부피 제약 mixture rest 용 activated scale
glm::vec4* d_rots = nullptr;     // 정규화 쿼터니언, PLY rot_0..3 순서 (SIBR 과 같은 규약)
float* d_outScales = nullptr;    // 변형된 모양 [N*3] (activated)
float* d_outQuats = nullptr;     // 변형된 방향 [N*4] (w, x, y, z)
float* d_opacity = nullptr;      // activated opacity [N] — 타원체 접촉의 k = √(2 ln(α/τ))
bool g_contactOn = false;        // 타원체 접촉 (xpbd_set_contact_shape)
float g_contactTau = 0.2f;
float g_contactCap = 0.0f;
int g_n = 0;
bool g_stepped = false;          // 첫 스텝에서 그래프가 GPU 에 올라간다
std::string g_err;

int fail(const std::string& msg)
{
	g_err = msg;
	fprintf(stderr, "[xpbd_dll] %s\n", msg.c_str());
	return -1;
}

int check_cuda(const char* where)
{
	const cudaError_t e = cudaGetLastError();
	if (e != cudaSuccess) return fail(std::string(where) + ": " + cudaGetErrorString(e));
	return 0;
}

void free_all()
{
	delete g_cm;
	g_cm = nullptr;
	if (d_scales) cudaFree(d_scales);
	if (d_rots) cudaFree(d_rots);
	if (d_outScales) cudaFree(d_outScales);
	if (d_outQuats) cudaFree(d_outQuats);
	if (d_opacity) cudaFree(d_opacity);
	d_opacity = nullptr;
	g_contactOn = false;
	FORWARD::setContactShape(false, false);
	d_scales = nullptr;
	d_rots = nullptr;
	d_outScales = nullptr;
	d_outQuats = nullptr;
	g_n = 0;
	g_stepped = false;
}
}  // namespace

XAPI const char* xpbd_last_error() { return g_err.c_str(); }

// n 개 가우시안과 무향 간선 e 개로 물리 상태를 만든다.
//  pos[n*3], scales[n*3] (activated), rots[n*4] (정규화, rot_0..3), opacity[n] (activated)
//  edges[e*2] (i, j), rest[e] (rest 거리), stiff[e] (간선 강성) — SIBR loadGraph 와 같은 입력
XAPI int xpbd_create(int n, const float* pos, const float* scales, const float* rots, const float* opacity,
	int e, const int* edges, const float* rest, const float* stiff)
{
	free_all();
	if (n <= 0 || !pos || !scales || !rots || !opacity || e < 0 || (e > 0 && (!edges || !rest || !stiff)))
		return fail("xpbd_create: invalid arguments");

	std::vector<FORWARD::Pos> p(n);
	for (int i = 0; i < n; ++i) p[i] = glm::vec3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
	std::vector<float> op(opacity, opacity + n);
	std::vector<FORWARD::Edge> es;
	es.reserve(e);
	for (int k = 0; k < e; ++k) es.emplace_back(edges[2 * k], edges[2 * k + 1], rest[k], stiff[k]);

	g_cm = new FORWARD::ChainMail();
	if (!g_cm->loadGraph(*g_cm, p, es, op) || (int)g_cm->numElements() != n) {
		free_all();
		return fail("xpbd_create: loadGraph failed");
	}
	cudaMalloc(&d_scales, sizeof(glm::vec3) * n);
	cudaMalloc(&d_rots, sizeof(glm::vec4) * n);
	cudaMemcpy(d_scales, scales, sizeof(glm::vec3) * n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rots, rots, sizeof(glm::vec4) * n, cudaMemcpyHostToDevice);
	cudaMalloc(&d_outScales, sizeof(float) * 3 * n);
	cudaMalloc(&d_outQuats, sizeof(float) * 4 * n);
	cudaMalloc(&d_opacity, sizeof(float) * n);
	cudaMemcpy(d_opacity, opacity, sizeof(float) * n, cudaMemcpyHostToDevice);
	FORWARD::setPhysicsMode(1);
	FORWARD::invalidatePhysicsGraph();   // 같은 크기의 이전 그래프가 GPU 에 남아 있어도 새로 올린다
	g_n = n;
	return check_cuda("xpbd_create");
}

XAPI void xpbd_destroy()
{
	FORWARD::setKinematicColliders(0, nullptr, nullptr, nullptr, 0.0f, 0.0f);   // 다음 데모에 충돌체가 남지 않게
	FORWARD::setAttachedParticles(0, nullptr, nullptr, 1.0f);
	free_all();
}

XAPI int xpbd_count() { return g_n; }

XAPI void xpbd_set_solver(int iters, float dt, float underRelax, float velDamping, float stiffnessScale,
	float invMassScale, float distCompliance, float shapeCompliance, float shapeBlend,
	float angleCompliance, float angleBlend)
{
	FORWARD::setXPBDParams(iters, dt, underRelax, velDamping, stiffnessScale, invMassScale,
		distCompliance, shapeCompliance, shapeBlend, angleCompliance, angleBlend);
}

XAPI void xpbd_set_constraints(int distance, int shape, int angle, int volume)
{
	FORWARD::setXPBDDistanceConstraintEnabled(distance != 0);
	FORWARD::setXPBDShapeMatchingEnabled(shape != 0);
	FORWARD::setXPBDAngleConstraintEnabled(angle != 0);
	FORWARD::setVolumeConstraintEnabled(volume != 0);
}

// 부피 제약: compliance, 클러스터 반경 k, 멤버 상한 cap, 리더 최소 간격 r (멤버 선발은 farthest-point)
XAPI void xpbd_set_volume(float compliance, int ringK, int maxMembers, int leaderMinHop)
{
	float c = 0.0f, aniso = 0.0f;
	FORWARD::getVolumeParams(&c, &aniso);
	FORWARD::setVolumeParams(compliance, aniso);
	FORWARD::setVolumeRingK(ringK, maxMembers);
	FORWARD::setVolumeLeaderParams(leaderMinHop, 1);
}

// 무한 평면 바닥 + 중력 (-up 방향). 모든 길이·가속도는 입력 좌표계 단위.
XAPI void xpbd_set_ground(int enabled, float ux, float uy, float uz, float height, float friction,
	float restitution, float contactRadius, float contactSlop, float gravity)
{
	const float up[3] = { ux, uy, uz };
	FORWARD::setGroundParams(enabled != 0, up, height, friction, restitution, contactRadius, contactSlop, gravity, false);
}

// 물체(그래프 연결 성분) 단위 형상 유지 강도. 0 = 끔.
XAPI void xpbd_set_object_shape(float stiffness) { FORWARD::setObjectModeParams(stiffness, false, 50); }

// 물체 단위 형상 유지의 누적·극분해: 1 = GPU (기본값), 0 = 기존 호스트 경로 (A/B 비교용).
XAPI void xpbd_set_object_shape_gpu(int enabled) { FORWARD::setObjectShapeGPU(enabled != 0); }

// 운동학 충돌체 (Isaac 물체 → 가우시안 한 방향). forward.h setKinematicColliders 참고. count 0 = 모두 끔.
XAPI void xpbd_set_colliders(int count, const int* types, const float* poses, const float* dims, float margin,
	float friction)
{
	FORWARD::setKinematicColliders(count, types, poses, dims, margin, friction);
}

// 마지막 스텝 충돌체별 (닿은 입자 수, 밀어낸 변위 합 xyz, 접촉 중심 xyz). 반환 = 채운 충돌체 수
XAPI int xpbd_collider_stats(int* hits, float* push, float* centroid, int maxCount)
{
	return FORWARD::getKinematicColliderStats(hits, push, centroid, maxCount);
}

// 모든 가우시안 속도에 강체 속도장 dv + dw × (x − center) 를 더한다 (입력 좌표계, 스텝 사이에). 물체 단위 충돌 충격량용.
XAPI int xpbd_add_rigid_velocity(const float* dv, const float* dw, const float* center)
{
	if (!g_stepped) return fail("xpbd_add_rigid_velocity: no step yet");
	FORWARD::addRigidVelocity(dv, dw, center);
	return check_cuda("xpbd_add_rigid_velocity");
}

// 타원체 접촉: 켜면 충돌체·바닥이 가우시안 중심이 아니라 불투명도 tau 등고면 타원체와 부딪힌다 (기본 끔).
//  enabled   : 비트 1 = 충돌체(로봇 손가락·던진 물체), 비트 2 = 바닥. 0 = 끔
//  tau       : 경계 불투명도 (가우시안 하나가 α·exp(−½d²) = tau 인 곳). 클수록 얇다. α ≤ tau 인 가우시안은 점.
//  radiusCap : 반축 상한 (입력 단위, 0 = 없음) — 튀는 큰 가우시안이 두꺼운 껍질을 만들지 않게.
// 켤 때는 rest 모양으로 묶고, 그 뒤로는 xpbd_compute_shapes 가 부를 때마다 변형된 모양으로 다시 묶는다.
XAPI int xpbd_set_contact_shape(int enabled, float tau, float radiusCap)
{
	if (!g_cm || !d_opacity) return fail("xpbd_set_contact_shape: call xpbd_create first");
	g_contactOn = enabled != 0;
	g_contactTau = tau;
	g_contactCap = radiusCap;
	if (g_contactOn &&
		!FORWARD::updateContactShapes(reinterpret_cast<const float*>(d_scales), reinterpret_cast<const float*>(d_rots),
			d_opacity, g_n, tau, radiusCap))
		return fail("xpbd_set_contact_shape: pack failed");
	FORWARD::setContactShape((enabled & 1) != 0, (enabled & 2) != 0);
	return check_cuda("xpbd_set_contact_shape");
}

// 외부 물체(로봇 손가락 등)가 붙잡은 가우시안: idx[count] 를 invMass 0 으로 고정하고 위치를 pos[count*3] (입력 좌표)로.
// 붙잡은 동안 스텝마다 새 위치로 다시 부른다. weight = 물체 단위 형상 유지의 강체 맞춤에서 붙잡힌 점 하나의 무게. count 0 = 놓기.
XAPI int xpbd_set_attached(int count, const int* idx, const float* pos, float weight)
{
	if (!g_stepped) return fail("xpbd_set_attached: no step yet");
	FORWARD::setAttachedParticles(count, idx, pos, weight);
	return check_cuda("xpbd_set_attached");
}

// 자기충돌 (SIBR 과 같은 규칙). withinBody=0 이면 다른 물체 쌍만, 다른 물체 경계상자 근처 입자만 본다.
XAPI void xpbd_set_self_collision(int enabled, float radiusScale, float excludeScale, int withinBody)
{
	FORWARD::setSelfCollisionParams(enabled != 0, radiusScale, excludeScale, withinBody != 0);
}

// 마지막 프레임 자기충돌 통계: out[0] 후보 수, out[1] 활성 입자 수, out[2] 상한 초과 교체 수
XAPI void xpbd_self_collision_stats(int* out)
{
	FORWARD::getSelfCollisionStats(nullptr, nullptr, &out[0], &out[2], &out[1]);
}

// 형상 제약의 회전 추출: 1 = double 극분해 (SIBR "robust rotation", 기본값, 느림), 0 = 기존 float32 경로.
XAPI void xpbd_set_shape_robust(int enabled) { FORWARD::setXPBDShapeRobustPolar(enabled != 0); }

// 물리 한 스텝 (dt 는 xpbd_set_solver 의 값). GPU 작업이 끝날 때까지 기다린다.
XAPI int xpbd_step()
{
	if (!g_cm) return fail("xpbd_step: call xpbd_create first");
	FORWARD::stepPhysicsOnly(*g_cm, d_scales, d_rots);
	g_stepped = true;
	cudaDeviceSynchronize();
	return check_cuda("xpbd_step");
}

// 현재 위치 디바이스 포인터 (float3[n]). 첫 스텝 전에는 0.
XAPI unsigned long long xpbd_positions_device()
{
	if (!g_stepped) return 0ull;
	return reinterpret_cast<unsigned long long>(FORWARD::getPhysicsPositionsDevice(nullptr));
}

// 현재 위치를 호스트로 복사 (out[n*3]).
XAPI int xpbd_positions_host(float* out, int n)
{
	if (!g_stepped) return fail("xpbd_positions_host: no step yet");
	int count = 0;
	const float* d = FORWARD::getPhysicsPositionsDevice(&count);
	if (!out || n != count || !d) return fail("xpbd_positions_host: size mismatch");
	cudaMemcpy(out, d, sizeof(float) * 3 * n, cudaMemcpyDeviceToHost);
	return check_cuda("xpbd_positions_host");
}

// 현재 위치로 변형된 가우시안 모양 (SIBR 렌더 경로와 같은 규칙). deformEps 이하로 움직인 가우시안은 원래 모양.
// hostScales[n*3] (activated), hostQuats[n*4] (w,x,y,z) 로 복사한다 (nullptr 이면 복사하지 않음).
// 반환값: 변형이 반영된 가우시안 수 (실패 시 -1).
XAPI int xpbd_compute_shapes(float deformEps, float* hostScales, float* hostQuats, int n)
{
	if (!g_stepped) return fail("xpbd_compute_shapes: no step yet");
	if (n != g_n) return fail("xpbd_compute_shapes: size mismatch");
	int deformed = 0;
	if (!FORWARD::computeDeformedShapes(d_scales, d_rots, d_outScales, d_outQuats, deformEps, &deformed))
		return fail("xpbd_compute_shapes: kernel failed");
	if (g_contactOn)   // 타원체 접촉은 보이는(변형된) 모양 그대로 — 다음 스텝부터 쓴다
		FORWARD::updateContactShapes(d_outScales, d_outQuats, d_opacity, n, g_contactTau, g_contactCap);
	if (hostScales) cudaMemcpy(hostScales, d_outScales, sizeof(float) * 3 * n, cudaMemcpyDeviceToHost);
	if (hostQuats) cudaMemcpy(hostQuats, d_outQuats, sizeof(float) * 4 * n, cudaMemcpyDeviceToHost);
	return check_cuda("xpbd_compute_shapes") == 0 ? deformed : -1;
}

// rest 로 되돌리고 속도 0.
XAPI int xpbd_reset()
{
	if (!g_stepped) return fail("xpbd_reset: no step yet");
	FORWARD::setAttachedParticles(0, nullptr, nullptr, 1.0f);   // 붙잡은 것을 먼저 놓는다
	FORWARD::squashReset();
	return check_cuda("xpbd_reset");
}

// rest 에서 출발: tilt(축×라디안) 만큼 무게중심 기준 회전 + 속도 v = lin + ang × (x − com).
XAPI int xpbd_launch(const float* lin, const float* ang, const float* tilt)
{
	if (!g_stepped) return fail("xpbd_launch: no step yet");
	return FORWARD::groundLaunch(lin, ang, tilt) ? check_cuda("xpbd_launch") : fail("xpbd_launch: groundLaunch failed");
}

// 프레스: 입력 좌표계 축(axis 0/1/2) 양쪽에 평판. fromLo=1 이면 min 쪽, 0 이면 max 쪽 평판이 rampPerSec [입력 단위/s] 로 다가온다.
// maxDispPct = 축 방향 크기 대비 최대 변위. 25/50/75/100% 에서 90프레임씩 멈춰 정착한다 (SIBR squash 하네스 그대로).
// 프레스 중에는 바닥·중력·물체 형상 유지가 적용되지 않는다 (squash 격리 규칙).
XAPI int xpbd_press_start(int axis, float rampPerSec, float maxDispPct, int fromLo)
{
	if (!g_stepped) return fail("xpbd_press_start: step once first");
	FORWARD::squashSetPress(true);
	FORWARD::squashSetPressFromLo(fromLo != 0);
	FORWARD::squashStart(axis, 0.08f, rampPerSec, maxDispPct);
	return FORWARD::squashIsActive() ? check_cuda("xpbd_press_start") : fail("xpbd_press_start: squashStart failed");
}

// 체크포인트 정착(25/50/75/100% 에서 90프레임 멈춤) on/off. 다음 xpbd_press_start 부터. 기본 on.
XAPI void xpbd_press_autolog(int enabled) { FORWARD::squashSetAutoLog(enabled != 0); }

// 평판을 떼어 낸다 (위치는 그대로 — 이후 스텝에서 탄성으로 돌아온다).
XAPI void xpbd_press_stop() { FORWARD::squashStop(); }

// 현재 평판 위치 lo/hi (축 좌표), 변위, 최대 변위. 반환값 1 = 프레스 진행 중.
XAPI int xpbd_press_state(float* lo, float* hi, float* curDisp, float* maxDisp)
{
	int axis = 0, top = 0, bot = 0;
	FORWARD::squashGetProgress(curDisp, maxDisp, &axis, &top, &bot);
	FORWARD::squashGetPlates(lo, hi);
	return FORWARD::squashIsActive() ? 1 : 0;
}

// 부피 가중 J (전역 부피비). enable 을 켜 두면 매 스텝 끝에 계산된다 (호스트로 내려받아 느리다 — 검증용).
XAPI void xpbd_volume_stats(int enable, float* jvw, float* jvwStd)
{
	FORWARD::setVolumeJStatsEnabled(enable != 0);
	FORWARD::getVolumeJVolumeWeighted(jvw, jvwStd);
}
