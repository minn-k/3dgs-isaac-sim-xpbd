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

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED
#include <cuda.h>
#include <vector>
#include <string>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	#pragma once


	// ===== ?먮즺援ъ“ =====
	struct Edge {
		int m_vert[2];
		float st;
		float rl;
		Edge(int v0, int v1, float restlen, float stiff = 1.0f)
			: st(stiff), rl(restlen) {
			m_vert[0] = v0; m_vert[1] = v1;
		}
	};

	typedef glm::vec3 Pos;

	template<int D>
	struct SHs
	{
		float shs[(D + 1) * (D + 1) * 3];
	};
	struct Scale
	{
		float scale[3];
	};
	struct Rot
	{
		float rot[4];
		float& operator[](int idx) { return rot[idx]; }
		float operator[](int idx) const { return rot[idx]; }
	};
	// GaussianView.cpp?먯꽌 ?щ∼+洹몃옒???앹꽦(踰≫꽣)???앸궡怨???踰덈쭔 ?몄텧
	/*void SetChainMailGraphFromVectors(
		const std::vector<Pos>& cropped_pos,
		const std::vector<Edge>& cropped_edges,
		const std::vector<float>& cropped_opacity
	);*/


	constexpr float AIR = 0.13f;
	constexpr float SKIN = 0.45f;
	constexpr float BONE = 0.75f;

	struct Vec3 {
		float x, y, z;
		Vec3() : x(0), y(0), z(0) {}
		Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

		Vec3 operator+(const Vec3& o) const { return Vec3(x + o.x, y + o.y, z + o.z); }
		Vec3 operator-(const Vec3& o) const { return Vec3(x - o.x, y - o.y, z - o.z); }
		Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }

		float length() const { return std::sqrt(x * x + y * y + z * z); }
		Vec3 normalized() const {
			float l = length();
			if (l < 1e-6f) return Vec3(0, 0, 0);
			return Vec3(x / l, y / l, z / l);
		}
	};

	struct CMConstraint {
		float dx, dy, dz;
		float xShearY, xShearZ;       // X諛⑺뼢 ?꾩튂?????Y, Z異??꾨떒
		float yShearX, yShearZ;       // Y諛⑺뼢 ?꾩튂?????X, Z異??꾨떒
		float zShearX, zShearY;       // Z諛⑺뼢 ?꾩튂?????X, Y異??꾨떒
		CMConstraint()
			: dx(0.01f), dy(0.01f), dz(0.01f),
			xShearY(0.01f), xShearZ(0.01f),
			yShearX(0.01f), yShearZ(0.01f),
			zShearX(0.01f), zShearY(0.01f)
		{}
		CMConstraint(float dx, float dy, float dz,
			float xSy, float xSz,
			float ySx, float ySz,
			float zSx, float zSy)
			: dx(dx), dy(dy), dz(dz),
			xShearY(xSy), xShearZ(xSz),
			yShearX(ySx), yShearZ(ySz),
			zShearX(zSx), zShearY(zSy) {}
	};
	struct float3x3 {
		float m[9]; // ?됰젹 ?먯냼瑜????곗꽑(row-major) ?먮뒗 ???곗꽑(column-major)?쇰줈 ??? ?먰븯??????쒖꽌??留욎떠 ?ъ슜
		__host__ __device__ float& operator()(int row, int col) { return m[row * 3 + col]; }
		__host__ __device__ float operator()(int row, int col) const { return m[row * 3 + col]; }
	};

	struct Neighbor {
		int idx;    // neighbor index
		float dist; // rest distance
		float st; // ?댁썐 媛?affinity
		float lambda; // XPBD ?쇨렇?묒＜ ?뱀닔(?쒖빟蹂?
		Neighbor() : idx(-1), dist(0), st(0), lambda(0.0f){}
		Neighbor(int idx, float dist, float st) : idx(idx), dist(dist), st(st), lambda(0.0f) {}
	};

	struct Element {
		glm::vec3 pos;
		glm::vec3 vel;
		float invMass;
		float density;
		float time;
		int offset;      // neighbor array start index
		int neighborCnt; // number of neighbors	
		Element(): pos(), vel(0.0f), invMass(1.0f), density(0), time(1e9f), offset(0), neighborCnt(0) {}
	};
	struct cEdge {
		int v1, v2;
		float dist;
	};
	struct SeedGroup {
		std::string name;          // 援щ텇???대쫫 (?? "LeftEar", "RightEar")
		std::vector<int> indices;  // ?먮뱾???몃뜳??紐⑸줉
		glm::vec3 pivot;           // ??洹몃９???뚯쟾 以묒떖??(留ㅼ슦 以묒슂!)

		// ?앹꽦??
		SeedGroup(std::string n, std::vector<int> idxs, glm::vec3 p)
			: name(n), indices(idxs), pivot(p) {}
	};
	
	class  ChainMail {
	public:
		// ChainMail ?대옒???대? (?먮뒗 ?ㅻ뜑)??異붽???蹂?섎뱾
		struct DeformTask {
			bool isRunning = false;
			int seedIdx = -1;
			glm::vec3 startPos;
			glm::vec3 targetPos;
			float progress = 0.0f;     // 0.0 ~ 1.0 (吏꾪뻾瑜?
			float speed = 2.0f;        // 1珥덉뿉 ?대룞??鍮꾩쑉 (2.0?대㈃ 0.5珥덈쭔???꾨떖)
		};

		DeformTask singleDeformTask; // 硫ㅻ쾭 蹂?섎줈 ?좎뼵
		bool FPS = false;
		bool loadGraph(
			ChainMail& cm,
			const std::vector<Pos>& cropped_pos,
			const std::vector<Edge>& cropped_edges,
			const std::vector<float>& cropped_opacity);
		void resetTime();
		void movePointPos(int* idx, const glm::vec3& dpos, std::vector<int>& activeSet);
		//void setPointPos(int idx, const glm::vec3& targetPos, std::vector<int>& activeSet);

		void propagate(std::vector<int>& activeSet); // propagated && moved 湲곗?
		void propagateStep(const std::vector<int>& currentFrontier, std::vector<int>& nextFrontier, std::vector<int>& totalActiveSet);
		void startWave(int seed, glm::vec3 delta, std::vector<int>& activeSet);
		void startWaveMultiple(const std::vector<int>& seeds, glm::vec3 delta, std::vector<int>& activeSet);
		void startWavingMultiple(
			const std::vector<int>& seeds,
			glm::vec3 delta,
			std::vector<int>& activeSet);
		std::vector<int> collectSeedsBFS(int startNode, int targetCount);
		void applyWaveOffset(const glm::vec3& delta, std::vector<int>& activeSet);

		void B_relax(const std::vector<int>& activeSet);
		void Stabilize(const std::vector<int>& activeSet);
		void relax(const std::vector<int>& activeSet);
		void relax2(const std::vector<int>& activeSet);

		size_t numElements() const;
		const Element& getElement(int i) const;
		Element& getElement(int i);
		const Neighbor& getNeighbor(int i) const;
		Neighbor& getNeighbor(int i);
		CMConstraint getConstraint(float);
		const std::vector<cEdge>& getEdges() const;
		std::vector<Edge> cropped_edges;
		float d_thresholdP = 0.0f;
		bool isWaveRunning() const { return waveRunning; }
	

		std::vector<int> getSeeds() const { return seeds; }
		bool getRunning() const { return waveRunning; }

		void setSeeds(const std::vector<int>& s) {
			seeds = s;
		}
		void setRunning(bool wR) {
			waveRunning = wR;
		}
		float _deformingPoint1;
		const std::vector<SeedGroup>& getSeedGroups() const { return seedGroups; }

		// [?섏젙] 洹몃９ 異붽? ?⑥닔 (洹 1媛?異붽????뚮쭏???몄텧)
		void addSeedGroup(const std::string& name, const std::vector<int>& indices, glm::vec3 pivot) {
			seedGroups.emplace_back(name, indices, pivot);
		}

		// [?섏젙] 珥덇린??
		void clearSeedGroups() {
			seedGroups.clear();
		}

	private:
		float propagationTime(const Element& e, const Element& n);
		void shiftElementPoint(Element& elem, const Element& n, float targDist, bool& moved);
		// ?곗씠??
		std::vector<Element> elements;
		std::vector<Neighbor> neighbors;
		std::vector<cEdge> cedges;
		std::vector<int> waveActive;
		std::vector<int> seeds;
		std::vector<SeedGroup> seedGroups; // <-- ?닿구濡??泥?
		bool waveRunning = false;


	};


	extern __constant__ float mytime[1];
	// Perform initial steps for each Gaussian prior to rasterization.

	void setChainmailActiveMapEnabled(bool enabled);
	float getChainmailActiveRatio();
	int getChainmailActiveCount();
	void setGpuChainmailMode(int mode); // 0: cpu, 1: gpu
	int getGpuChainmailMode();
	void setPhysicsMode(int mode); // 0: ChainMail, 1: XPBD
	int getPhysicsMode();
	void setXPBDDistanceConstraintEnabled(bool enabled);
	bool getXPBDDistanceConstraintEnabled();
	void setXPBDShapeMatchingEnabled(bool enabled);
	bool getXPBDShapeMatchingEnabled();
	// 형상 제약 회전 추출: true = double + 상대 멈춤 기준(기본 — 가는 부위가 정지 상태에서도 도는 문제 수정),
	// false = 기존 float32 경로 (A/B 비교용)
	void setXPBDShapeRobustPolar(bool enabled);
	bool getXPBDShapeRobustPolar();
	void setXPBDAngleConstraintEnabled(bool enabled);
	bool getXPBDAngleConstraintEnabled();

	// Volume Gaussian (부피 가우시안)
	void setVolumeConstraintEnabled(bool enabled);
	bool getVolumeConstraintEnabled();
	// Independent experimental A/B solver: best-fit affine F on the existing
	// Gaussian volume clusters, with Stable Neo-Hookean hydrostatic and
	// distortional XPBD constraints. It does not replace or mutate legacy paths.
	void setGaussianNHEnabled(bool enabled);
	bool getGaussianNHEnabled();
	// 거리 제약을 간선 컬러링 Gauss-Seidel 로 푼다 (기본 OFF = 기존 Jacobi)
	void setDistanceGSEnabled(bool enabled);
	bool getDistanceGSEnabled();
	void getDistanceGSStats(int* numEdges, int* numColors, int* oneWayEdges);
	// 부피 제약을 클러스터 컬러링 Gauss-Seidel 로 푼다 (기본 OFF = 결정론적 Jacobi gather)
	void setVolumeGSEnabled(bool enabled);
	bool getVolumeGSEnabled();
	void getVolumeGSStats(int* numClusters, int* numColors, int* lowerBound, bool* skipped);
	// 솔버 반복 루프(모든 제약 × iters) 시간. 켜면 substep 마다 동기화 1회 — 비교 실험용
	void setSolverTiming(bool enabled);
	void getSolverTiming(float* avgMs, int* samples, bool reset);
	void setGaussianNHMaterial(float young, float poisson, float complianceScale);
	void getGaussianNHMaterial(float* young, float* poisson, float* complianceScale);
	void setVolumeParams(float compliance, float anisoThreshold);
	void getVolumeParams(float* compliance, float* anisoThreshold);
	void getVolumeMatTypeCounts(int* volumeCnt, int* surfaceCnt, int* fiberCnt);

	// k-ring 클러스터 반경 (k=1: 그래프 직접 이웃 = 기존 동작, k>1: BFS k-hop 확장)
	// maxMembers: 클러스터당 멤버 상한 (BFS 결과를 균등 서브샘플; 비용/메모리 상한 고정)
	void setVolumeRingK(int k, int maxMembers);
	void getVolumeRingK(int* k, int* maxMembers);

	// 실행 전략: 0 = scatter (atomicAdd 산란), 1 = gather (전치 CSR 읽기, atomic 없음/결정론적)
	// 두 모드는 수학적으로 동일한 계산이다. 성능/재현성 비교용.
	void setVolumeGatherMode(int mode);
	int getVolumeGatherMode();

	// 반장 희소화(Poisson-disk 최소 간격 minHop) + 반원 선발(0=stride, 1=farthest-point)
	// minHop=1이면 모든 노드가 반장(기존). 자유도 N개는 불변, 부피 측정소 M개만 줄어든다.
	void setVolumeLeaderParams(int minHop, int memberSelect);
	void getVolumeLeaderInfo(int* minHop, int* memberSelect, int* leaderCount);
	// h_j 상한 U (양방향 유계 커버). 멤버 선발에서 이미 U겹 덮인 후보를 제외한다.
	// 기존 greedy는 하한만 보장해 인기 노드가 수백~수천 겹이 됐고(k=3에서 max 937),
	// 그 꼬리가 gather 커널의 워프 대기를 만들어 비용을 지배했다.
	// 0 = 비활성(기존 동작). 켜면 리빌드가 순차로 돈다(coverCnt를 순서대로 봐야 하므로).
	void setVolumeCoverMax(int coverMax);
	int  getVolumeCoverMax();

	// Σ⁻¹ 축퇴 방어 방식 (A/B 토글).
	//  false = 고유분해 후 최소축을 최대축의 6%로 클램프 (기존 G4). 반복 Jacobi라 ~600 FLOP.
	//  true  = Σ + eps·(trace/3)·I 로 대각선을 들어올림. ~15 FLOP. detHat<detThr 인 축퇴 클러스터에만 적용.
	// 이 구간은 멤버 수 n과 무관한 '직렬 구간'이라 클러스터당 병렬화(워프/블록)의 Amdahl 병목이 된다.
	void setVolumeSinvIsoReg(bool enabled, float eps, float detThr);
	void setVolumeCovReg(bool enabled, float lambda);   // 렌더 변형 F: 공분산 쿠션(젤리) 토글
	void requestJSmoothnessLog(const char* tag);        // [진단] per-Gaussian J 집계 매끄러움 1회 측정
	// [TN] 접선-법선 분해: 접선 2D는 최소제곱, 법선 1자유도는 물리 J로 닫는 렌더 변형 경로
	void setTangentNormalMode(bool enabled, float flatRatio);
	void getTangentNormalStats(unsigned int* applied, unsigned int* fallback);
	void resetTangentNormalStats();
	// 렌더 F 축퇴 강건화: trace 정규화 det 판정 + PPt/QPt 양쪽 정규화.
	// OFF 면 기존 경로(얇은 구조가 회전을 아예 못 함) 그대로라 A/B 가 된다.
	void setRenderFRobust(bool enabled);
	bool getRenderFRobust();
	void getRenderFStats(unsigned int* applied, unsigned int* detFail, unsigned int* fewNbr);

	// ── [SH] 변형 회전을 겉모습(구면조화)에 반영 ────────────────────
	//  0 = OFF(기존) · 1 = ON(dir <- R^T dir) · 2 = 진단용 역방향
	//  모양 경로와 독립. OFF 면 기존 결과와 비트 단위로 동일하다.
	void setSHRotate(int mode);
	int  getSHRotate();
	void getSHRotStats(unsigned int* applied, unsigned int* fallback);
	void resetSHRotStats();
	void resetRenderFStats();
	void getVolumeSinvIsoReg(bool* enabled, float* eps, float* detThr);

	// Kernel A/B 실행 형태 (gather 모드 전용). 0=자동(기존 캐시형 워프) · 1=스레드당
	// · 2=캐시형 워프당 · 3=1-pass 모멘트 워프당. 모드 3은 A만 바꾸고 B는 기존 워프를 쓴다.
	// ⚠️ 워프가 항상 빠르지 않다 — 클러스터 멤버 수 n에 달렸다. 측정(scene, r=1):
	//     k=1 (n=25) : 6.55 → 8.73 ms  (33% 느려짐)   |  k=3 (n=64) : 20.0 → 13.5 ms (1.48배)
	//   n이 작으면 shuffle 오버헤드가 실제 계산을 넘어선다. 자동 모드는 리빌드 때 잰
	//   '클러스터당 평균 멤버 수'가 40 이상일 때만 워프를 쓴다.
	void  setVolumeWarpMode(int mode);
	int   getVolumeWarpMode();
	float getVolumeAvgMembers();   // 마지막 리빌드의 클러스터당 평균 멤버 수 (자동 판단 근거)

	// Step 3 — 물성 앵커: α_vol(i) = c_vol / (λ_Lamé(E,ν) · V_rest(i))
	// enabled=false면 기존 손 튜닝 상수(compliance) 사용. c_vol=1이 이론값.
	void setVolumePhysicalMaterial(bool enabled, float E, float nu, float cvol);
	void getVolumePhysicalMaterial(bool* enabled, float* E, float* nu, float* cvol);

	// mixture rest: V_rest(명찰)와 matType(신분증)에 가우시안 자신의 모양(R·S²·Rᵀ)을
	// opacity 가중으로 반영. 런타임 J 측정(저울)은 불변 — 정확 등식 유지.
	void setVolumeMixtureRest(bool enabled);
	bool getVolumeMixtureRest();

	// Step 2 — J 시각화: SH 색 대신 J 컬러맵으로 렌더 (파랑 J<1 / 흰색 1 / 빨강 J>1)
	void setVolumeJVizEnabled(bool enabled);
	bool getVolumeJVizEnabled();
	void setVolumeJVizGain(float gain);   // (J-1)*gain 이 ±1에서 색 포화. 기본 3.0
	float getVolumeJVizGain();

	// matType 시각화: 물질 분류를 색으로 (volume=회색 / surface=파랑 / fiber=빨강)
	// 모양 기반 분류가 실제 재질과 맞는지 눈으로 검증하는 용도
	void setVolumeMatVizEnabled(bool enabled);
	bool getVolumeMatVizEnabled();

	// J 통계 (매 프레임 자동 계산; 볼륨 제약이 실제로 J를 1로 되돌리는지 확인)
	void setVolumeJStatsEnabled(bool enabled);
	bool getVolumeJStatsEnabled();
	void getVolumeJStats(int* count, float* mean, float* std, float* jmin, float* jmax, float* p05, float* p95);
	// 부피 가중 J = Σ V_rest·J / Σ V_rest. 위의 '개수 평균'과 달리 큰 덩어리에 큰 표를 준다.
	// 전역 부피 보존 오차의 정직한 지표. 둘이 크게 다르면 작은 클러스터가 통계를 지배한다는 뜻.
	void getVolumeJVolumeWeighted(float* vwMean, float* vwStd);
	// 현재 J 통계를 콘솔에 한 줄로 기록 (k·cap·r·M 조건 포함). k 스윕 기록용.
	void logVolumeJStats(const char* tag);

	// Squash Test (위 슬랩을 아래 슬랩 쪽으로 눌러 옆으로 삐져나오는지 관찰)
	void squashStart(int axis, float slabPct, float rampPerSec, float maxDispPct);
	void squashStop();
	void squashReset();
	bool squashIsActive();
	// 완료된 체크포인트 수 (0~4). 자동 스윕의 '이번 실행 끝' 신호.
	int squashCheckpointsDone();
	// 평판 경계조건: true = 미끄럼(하중축만 구속), false = no-slip(세 축 구속).
	void squashSetSlip(bool slip);
	// true = 슬랩 핀 대신 위·아래 평판을 바닥과 같은 규칙(밖으로 나간 중심만 투영, 마찰 없음)으로 건다. 다음 Start 부터.
	void squashSetPress(bool press);
	// press 모드에서 움직일 평판: false = max 쪽(기본), true = min 쪽. 좌표계의 '위'가 −축인 데이터에서 위 평판을 누르게 할 때.
	void squashSetPressFromLo(bool fromLo);
	// 체크포인트 정착(25/50/75/100% 에서 멈춤 + J 기록) on/off. 기본 on (FEM 비교 규약).
	void squashSetAutoLog(bool enabled);
	// 현재 평판 위치 (squash 축 좌표). press 가 아니면 rest 경계.
	void squashGetPlates(float* lo, float* hi);
	bool squashGetSlip();
	void squashGetProgress(float* curDisp, float* maxDisp, int* axis, int* topCnt, int* botCnt);

	// ── Ground / drop demo (XPBD 전용) ──────────────────────────────────────
	// 무한 평면 바닥 1장 + 중력. 가우시안 중심을 반경 contactRadius 인 구로 보고
	// n·x ≥ height + contactRadius 로 투영한다. 입자쌍 검사가 없어 O(N).
	//  up          : 바닥 법선 (내부에서 정규화). 평면은 n·x = height.
	//  friction    : 쿨롱 계수 μ (충격량 형태).  restitution: 반발 계수 e ∈ [0,1].
	//  contactSlop : 속도 단계에서 '접촉 중'으로 볼 gap 여유 (world).
	//  gravity     : world units/s², −up 방향. 0이면 중력 없음. 데이터셋 단위가 미터가 아니므로
	//                호출 측이 물체 높이로 정규화해서 넘긴다.
	//  paused      : true면 XPBD 스텝을 건너뛴다 (렌더는 계속).
	void setGroundParams(bool enabled, const float up[3], float height, float friction, float restitution,
		float contactRadius, float contactSlop, float gravity, bool paused);
	// 체커 바닥 렌더 (배경 대신 픽셀 광선-평면 교점을 합성). 매 프레임 카메라와 함께 호출한다.
	//  viewmatrix : 래스터라이저에 넘기는 것과 같은 열우선 4x4 (행 1·2 반전 규약)
	void setGroundRender(bool visible, const float origin[3], float checkerSize, float fadeRadius,
		const float* viewmatrix, const float campos[3], float tanFovX, float tanFovY);
	// rest로 되돌린 뒤 rest 무게중심 기준으로 tiltAxisAngle(축×라디안)만큼 회전시키고,
	// 모든 가우시안에 v = linVel + ω × (x − com) 을 준다. 그래프가 아직 없으면 false.
	bool groundLaunch(const float linVel[3], const float angVel[3], const float tiltAxisAngle[3]);

	// Isaac Sim 연동: GPU 에 올라간 물리 그래프(CSR + rest 위치)를 뷰어 내부 순서 그대로 바이너리로 쓴다.
	// 형식과 USD 순서 변환은 tools/import_graph.py 참조. 그래프가 아직 GPU 에 없으면 false.
	bool exportPhysicsGraph(const char* path);

	// 외부 호스트(Isaac Sim DLL)용: 렌더 없이 XPBD 물리 한 스텝. 파라미터는 기존 setter 들(setXPBDParams 등)로 넣는다.
	//  scales    : activated scale (glm::vec3[N]) 디바이스 포인터 — 부피 제약 mixture rest 용
	//  rotations : 정규화 쿼터니언 (glm::vec4[N], PLY rot_0..3 순서) 디바이스 포인터
	void stepPhysicsOnly(ChainMail& cm, const glm::vec3* scales, const glm::vec4* rotations);
	// 현재 위치 디바이스 버퍼 (float3[N]). count 에 N.
	const float* getPhysicsPositionsDevice(int* count);
	// 다음 스텝에서 그래프·버퍼를 다시 올리게 한다 (같은 크기의 다른 그래프를 불러올 때).
	void invalidatePhysicsGraph();
	// 외부 호스트용: 현재 위치로 변형된 가우시안 모양(축 크기, 쿼터니언 w,x,y,z)을 렌더 경로와 같은 규칙으로 계산.
	//  출력은 디바이스 버퍼 outScales[N*3] (activated), outQuats[N*4]. deformEps 이하로 움직인 가우시안은 원래 모양.
	bool computeDeformedShapes(const glm::vec3* scales, const glm::vec4* rotations,
		float* outScales, float* outQuats, float deformEps, int* deformedCount);
	// 물체 단위 형상 유지(전역 shape matching, predict 직후 1회) + 물체 단위 바닥 반발.
	//  shapeStiffness : 0 = off, 1 = 매 프레임 rest 모양으로 완전히 되돌림 (회전·이동은 자유)
	//  bodyContact    : 반발 충격량을 물체 전체 유효질량으로 계산 (e는 setGroundParams의 restitution).
	//                   켜면 입자 단위 반발은 0으로 둔다 (마찰은 입자 단위 유지)
	//  minComponent   : 물체 = 그래프 연결 성분. 이 크기 미만 성분(floater)은 물체로 보지 않는다.
	//                   여러 물체가 있는 장면(pillow)에서 물체마다 따로 형상 유지·반발한다.
	void setObjectModeParams(float shapeStiffness, bool bodyContact, int minComponent);
	// 물체 단위 형상 유지의 누적·극분해를 GPU 에서 (기본 ON). false = 기존 호스트 경로 (A/B 비교용).
	// 물체가 64개를 넘으면 설정과 무관하게 호스트 경로.
	void setObjectShapeGPU(bool enabled);
	bool getObjectShapeGPU();
	// 운동학 충돌체 (외부 물체 → 가우시안 한 방향). 입력 좌표계 값, 매 프레임 다시 줘도 된다 (최대 32개).
	//  types : 0 sphere, 1 box, 2 capsule
	//  poses : 충돌체마다 12개 = 로컬→입력 회전 R(행우선 9) + 중심 t(3)
	//  dims  : 충돌체마다 3개 = box 반 크기 / sphere (반지름,-,-) / capsule (반지름, 반 길이(로컬 z), -)
	//  margin  : 입자를 표면에서 이만큼 밖으로 둔다 (입력 단위)
	//  friction: 충돌체 표면 마찰 μ (위치 단계, 0 = 없음). 같은 목록으로 다시 부르면 직전 자세를 기억해 표면이 움직인
	//            만큼은 입자를 끌고 간다
	void setKinematicColliders(int count, const int* types, const float* poses, const float* dims, float margin,
		float friction);
	// 마지막 스텝의 충돌체별 통계: hits = 마지막 반복에서 닿은 입자 수, push = 전체 반복에서 밀어낸 변위 합 (×3),
	// centroid = 밀어낸 크기로 가중한 접촉 위치 (×3, 접촉이 없으면 충돌체 중심). 반작용(양방향 결합)에 쓴다.
	int getKinematicColliderStats(int* hits, float* push, float* centroid, int maxCount);
	// 모든 입자 속도에 강체 속도장 dv + dw × (x − center) 를 더한다 (입력 좌표계). 외부 강체와 부딪힌 순간의
	// 물체 단위 충격량을 가우시안 물체에 줄 때 쓴다 (스텝과 스텝 사이에 부른다).
	void addRigidVelocity(const float dv[3], const float dw[3], const float center[3]);
	// 외부 물체(로봇 손가락 등)가 붙잡은 가우시안: idx[count] 를 invMass 0 으로 고정하고 위치를 pos[count*3] 로 둔다
	// (입력 좌표, 스텝 사이에 부른다 — 붙잡은 동안 스텝마다 새 위치로). weight = 물체 단위 형상 유지의 강체 맞춤에서
	// 붙잡힌 점 하나의 무게 (나머지 1). count 0 = 놓기 (invMass 를 붙잡기 전 값으로).
	void setAttachedParticles(int count, const int* idx, const float* pos, float weight);
	// 타원체 접촉 (기본 OFF): 켜면 충돌체·바닥 판정에서 가우시안을 불투명도 τ 등고면 타원체로 본다.
	// updateContactShapes: 디바이스 포인터 scales[n*3] (activated), quats[n*4] (w,x,y,z), opacity[n] (activated) 로
	// 타원체를 묶는다 (모양이 바뀔 때마다 — Isaac 은 프레임마다 변형 모양 계산 뒤에). radiusCap > 0 이면 반축 상한 (입력 단위).
	void setContactShape(bool colliders, bool ground);   // 충돌체 / 바닥 각각
	bool updateContactShapes(const float* scales, const float* quats, const float* opacity, int n, float tau,
		float radiusCap);
	// 마지막으로 나눈 물체 통계. largest 는 크기 상위 3개(int[3])
	// ⚠️ largest 는 int[3] 이다 (상위 3개 성분 크기). int 하나를 넘기면 호출자 스택을 덮어쓴다.
	void getObjectComponentStats(int* objects, int largest[3], int* smallComponents, int* smallParticles);
	// 자기충돌 ('처음 붙어 있던 쌍 제외'): rest 거리 < excludeScale·d_c 인 쌍은 충돌하지 않는다.
	//  radiusScale : 접촉 거리 d_c = radiusScale × 간격(그래프 최소 엣지 길이 중앙값)
	//  excludeScale: d_ex = excludeScale × d_c (≥ 1.05로 제한 — 이하면 rest에서 겹친 쌍이 들어와 터진다)
	//  withinBody  : 같은 물체 안 접촉(잎끼리·머리카락)까지 본다. 끄면 다른 물체 쌍만 보고,
	//                다른 물체 경계상자 근처 입자만 격자에 넣어 훨씬 빠르다 (Keep object shape 가 켜져 있을 때).
	void setSelfCollisionParams(bool enabled, float radiusScale, float excludeScale, bool withinBody);
	void getSelfCollisionStats(float* spacing, float* contactDist, int* contacts, int* overflow, int* active);

	// FEM reference benchmark: opt-in CSV dump at settled squash checkpoints.
	void setFEMBenchmarkDumpEnabled(bool enabled, const std::string& directory, const std::string& runLabel);
	bool getFEMBenchmarkDumpEnabled();

	// Region Balloon (선택 BFS 영역을 하나의 전역 부피 타원체로 취급)
	void setRegionBalloonEnabled(bool enabled);
	bool getRegionBalloonEnabled();
	void setRegionBalloonParams(float compliance, float strength, float maxStep, int hops);
	void getRegionBalloonParams(float* compliance, float* strength, float* maxStep, int* hops);
	int getRegionBalloonCount();

	void setPickingParams(int hops);
	void getPickingParams(int* hops);
	bool copyCurrentDeformedPositions(float* outXYZ, int pointCount);
	bool copyCurrentScreenProjection(
		float* outXY,
		int* outRadii,
		int pointCount,
		int* outRenderW,
		int* outRenderH);

	void setChainmailParams(
		int propIters,
		int relaxIters,
		float propStrength,
		float stiffness,
		float damping);
	void getChainmailParams(
		int* propIters,
		int* relaxIters,
		float* propStrength,
		float* stiffness,
		float* damping);
	void setChainmailMaterialParams(
		float constraintGlobalScale,
		float airScale,
		float skinScale,
		float boneScale,
		bool useEdgeStiffness,
		float edgeStiffnessInfluence);
	void getChainmailMaterialParams(
		float* constraintGlobalScale,
		float* airScale,
		float* skinScale,
		float* boneScale,
		bool* useEdgeStiffness,
		float* edgeStiffnessInfluence);
	void setChainmailDynamicsParams(
		float inertiaGain,
		float velocityRetention,
		float velocityClamp);
	void getChainmailDynamicsParams(
		float* inertiaGain,
		float* velocityRetention,
		float* velocityClamp);
	void setXPBDParams(
		int solverIters,
		float dt,
		float underRelax,
		float velDamping,
		float stiffnessScale,
		float invMassScale,
		float distanceCompliance,
		float shapeCompliance,
		float shapeBlend,
		float angleCompliance,
		float angleBlend);
	void getXPBDParams(
		int* solverIters,
		float* dt,
		float* underRelax,
		float* velDamping,
		float* stiffnessScale,
		float* invMassScale,
		float* distanceCompliance,
		float* shapeCompliance,
		float* shapeBlend,
		float* angleCompliance,
		float* angleBlend);

	void preprocess(
		ChainMail& cm, std::vector<int>& activeSet,

		int P, int D, int M,
		 float* orig_points,
		int nbr_K,

		const glm::vec3* scales,
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


		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		bool* clamped,
		const float* cov3D_precomp,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* points_xy_image,
		float* depths,
		float* cov3Ds,
		float* colors,
		float4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered,
		int2* rects,
		float3 boxmin,
		float3 boxmax,
		bool antialiasing,
		float t,
		bool _wave,
		bool _twist,
		bool _bubble);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* points_xy_image,
		const float* features,
		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color,
		int* id_buffer);

	
	// ?몃??먯꽌 ?묎렐???꾩뿭 ChainMail ?몄뒪?댁뒪
	/*extern ChainMail g_chainmail;
	extern bool      g_chainmail_ready;*/
}


#endif


