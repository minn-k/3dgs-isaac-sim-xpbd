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
#include<cuda_fp16.h>
#include <cuda_runtime.h>
// 자기충돌 공간 해시 정렬. 프로젝트 헤더보다 먼저 — cub 내부 템플릿 인자 NUM_CHANNELS가 매크로와 충돌한다.
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>    // 자기충돌: 프레임 최대 변위 (격자 칸 크기)
#include "header.h"
#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <queue>
#include <set>
#include <fstream>
#include <iomanip>
#include <cctype>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
namespace cg = cooperative_groups;
static float t = 0;

// GPU command mode: CPU only submits (idx, delta) commands, GPU applies them.
static bool g_gpuCommandMode = false;
static std::vector<int> g_cmdIdx;
static std::vector<glm::vec3> g_cmdDelta;
static int bfsHops = 2;
// Chainmail active map mode (GPU)
static int g_GpuChainmailMode = 1;
static bool g_useActiveMap = false;
static bool g_activeMapReset = false;
static int g_lastActiveCount = 0;
static float g_lastActiveRatio = 0.0f;
// 0: ChainMail, 1: XPBD
static int g_physicsMode = 0;
static bool g_useXPBDDistanceConstraint = true;
static bool g_useXPBDShapeMatching = false;
static bool g_useXPBDAngleConstraint = false;
// 형상 제약 회전 추출: true = 3×3 부분 double + 상대 멈춤 기준(기본), false = 기존 float32 경로(A/B용)
static bool g_xpbdShapeRobustPolar = true;

// ChainMail tunables (UI controllable)
static int g_cmPropIters = 20;
static int g_cmRelaxIters = 5;
static float g_cmPropStrength = 0.5f;
static float g_cmStiffness = 0.99f;
static float g_cmDamping = 0.1f;
// ChainMail material controls (defaults preserve legacy behavior).
static float g_cmConstraintGlobalScale = 1.0f;
static float g_cmConstraintAirScale = 0.1f;
static float g_cmConstraintSkinScale = 0.1f;
static float g_cmConstraintBoneScale = 0.1f;
static bool g_cmUseEdgeStiffness = true;
static float g_cmEdgeStiffnessInfluence = 1.0f;
// Optional motion controls for cloth-like flutter effects.
static float g_cmInertiaGain = 0.0f;
static float g_cmVelocityRetention = 0.92f;
static float g_cmVelocityClamp = 0.05f;

// XPBD tunables (UI controllable)
static int g_xpbdSolverIters = 20;
static float g_xpbdDt = 1.0f / 60.0f;
static float g_xpbdUnderRelax = 0.6f;
static float g_xpbdVelDamping = 0.59f;
// Runtime global scales exposed in UI.
// stiffnessScale affects graph edge stiffness (nbrStiff), invMassScale affects all XPBD constraint weights.
static float g_xpbdStiffnessScale = 1.0f;
static float g_xpbdInvMassScale = 1.0f;
static float g_xpbdgravityScale = 1.0f;
static float g_xpbdDistanceCompliance = 1e-3f;
static float g_xpbdShapeCompliance = 0.0f;
static float g_xpbdShapeBlend = 0.1f;
// Angle constraint is sensitive: start soft and conservative by default.
static float g_xpbdAngleCompliance = 0.0f;
static float g_xpbdAngleBlend = 0.2f;

// ── Ground / drop demo: 무한 평면 바닥 + 중력 ─────────────────────────────
// 가우시안 중심을 반경 r 인 구로 보고 평면 하나와만 접촉시킨다.
//   C(x) = n·x − (h + r) ≥ 0      n: 단위 up,  평면은 n·x = h
// 입자쌍 검사가 없어 O(N)이고 입자마다 독립이라 Jacobi/GS 차이도 없다.
// 반복마다 내적 1번짜리 커널이 하나 늘 뿐이며, OFF면 커널을 띄우지 않아 기존 경로와 비트 단위로 같다.
// 마찰·반발은 위치 단계가 아니라 속도 갱신 단계에서 충격량 형태로 처리한다 (Müller et al. 2020 §3.6).
static bool  g_groundEnabled = false;
static bool  g_groundPaused = false;
static float g_groundN[3] = { 0.0f, 0.0f, 1.0f };
static float g_groundHeight = 0.0f;
static float g_groundFriction = 0.5f;
static float g_groundRestitution = 0.3f;
static float g_groundRadius = 0.0f;
static float g_groundSlop = 0.0f;      // 속도 단계에서 '접촉 중'으로 볼 gap 여유 (world)
// 충돌 관찰 로그: Launch(Drop/Throw) 뒤 이 프레임 수만큼 형상 지표를 찍는다 (Press 체크포인트는 정착 후라 과도응답을 못 본다).
static int g_impactLogLeft = 0;
static int g_impactLogFrame = 0;
static float g_groundGravity = 0.0f;   // world units/s², −n 방향. 0 = 중력 없음
// 체커 바닥 렌더: renderCUDA가 배경색 자리에 픽셀 광선-평면 교점의 체커를 합성한다.
static bool  g_groundVisible = false;
static float g_groundOrigin[3] = { 0.0f, 0.0f, 0.0f };
static float g_groundChecker = 1.0f;
static float g_groundFadeRadius = 10.0f;
static float g_groundViewM[16] = { 0.0f };
static float g_groundCamPos[3] = { 0.0f, 0.0f, 0.0f };
static float g_groundTanFovX = 1.0f;
static float g_groundTanFovY = 1.0f;

// ── 운동학 충돌체 (외부 엔진의 물체: Isaac 큐브·로봇 링크 등) ──────────────────────────────────────
// 한 방향: 충돌체가 가우시안을 밀어낼 뿐, 가우시안은 충돌체를 밀지 않는다 (충돌체 자세는 매 프레임 밖에서 준다).
// 바닥과 같은 규칙 — 매 반복의 마지막(바닥 직전)에 입자를 표면 + margin 밖으로 투영한다.
// 마찰은 위치 단계에서 (Macklin et al. 2014 §6.1): 접촉 입자의 이번 스텝 변위 중 충돌체 표면에 대한 접선 성분을
// 밀어낸 깊이 × μ 까지 되돌린다. 충돌체 표면의 움직임(직전 자세 → 지금 자세)은 빼고 본다 — 움직이는 충돌체가 끌고 간다.
// 좌표는 입력 좌표계(물리와 같은 원본 3DGS 좌표). 반작용 계산용으로 스텝마다 충돌체별 밀어낸 변위 합을 모은다.
#define MAX_KIN_COLLIDERS 32
struct KinCollider {
	float R[9];    // 충돌체 로컬 → 입력 좌표 회전 (행우선)
	float t[3];    // 중심
	float h[3];    // box: 반 크기(로컬 x,y,z) | sphere: h[0] 반지름 | capsule: h[0] 반지름, h[1] 반 길이(로컬 z 축)
	int type;      // 0 sphere, 1 box, 2 capsule
	float Rp[9];   // 직전 스텝 자세 (마찰에서 표면이 움직인 양을 뺄 때). 처음 들어온 충돌체는 지금 자세와 같다
	float tp[3];
};
static KinCollider g_kinCol[MAX_KIN_COLLIDERS];
static int    g_kinColCount = 0;
static float  g_kinColMargin = 0.0f;
static float  g_kinColFriction = 0.0f;
static bool   g_kinColDirty = false;
static KinCollider* d_kinCol = nullptr;
static int*   d_kinColHits = nullptr;      // [MAX] 마지막 스텝의 마지막 반복에서 닿은 입자 수
static float* d_kinColPush = nullptr;      // [MAX×7] 마지막 스텝 전체 반복: 밀어낸 변위 합(3), |변위| 가중 위치 합(3), |변위| 합(1)
#define KINCOL_ACC 7

// ── 타원체 접촉 (기본 OFF — 뷰어·논문 실험은 비트 단위로 예전과 같다) ──────────────────────────────────
// 가우시안을 중심점이 아니라 '불투명도가 τ 로 떨어지는 등고면' 타원체로 충돌체·바닥에 부딪힌다.
//   α·exp(−½ d_M²) = τ  →  k = √(2 ln(α/τ))  (α ≤ τ 이면 k = 0, 즉 점).   M = k²·Σ = R diag((k s)²) Rᵀ
//   법선 n 쪽으로 튀어나온 길이(지지 거리) r(n) = √(nᵀ M n) — 평면에서는 정확, 캡슐·상자는 가장 가까운 점 법선 기준 근사.
// Σ 는 렌더와 같은 변형 모양 (Isaac 은 프레임마다 xpbd_compute_shapes 뒤에 다시 묶는다 → 한 프레임 전 모양).
// 비용: 충돌체 판정 전 입자당 반경 상한 rmax 로 조기 탈락 (+4 B), 가까운 입자만 M 을 읽는다. 바닥은 스텝마다
//       법선 방향 반경을 미리 계산해 (+4 B/반복) 쓴다.
static bool   g_contactShapeOn = false;      // 충돌체(로봇 손가락·던진 물체)
static bool   g_contactShapeGround = false;  // 바닥 — 실측: 켜면 큰 반투명 가우시안이 몸을 늘인다 (dense-model long edges 0.04% → 1~10%)
static int    g_contactN = 0;
static float* d_contactM = nullptr;        // [N×6] k²Σ (xx, xy, xz, yy, yz, zz), 입력 좌표
static float* d_contactRmax = nullptr;     // [N] 가장 긴 반축 k·s_max
static float* d_contactRg = nullptr;       // [N] 바닥 법선 방향 반경 √(nᵀMn) (스텝마다)
__global__ void contactShapePackKernel(int N, const float* sc, const float* qt, const float* op, float tau, float rcap,
	float* M, float* rmax);

// ── Object mode: 물체 단위 형상 유지 + 물체 단위 반발 (runXPBDSimulation 호스트 단계) ──
static float  g_objShapeStiffness = 0.0f;  // 0 = off. predict 직후 전역 shape matching 강도
static bool   g_objBodyContact = false;    // 바닥 반발을 물체 전체 유효질량으로 계산
static bool   g_objPrevValid = false;      // 직전 프레임 강체 속도 기록이 유효한가 (launch·설정 변경 시 false)
static std::vector<float3> g_objHostPos;   // 다운로드 재사용 버퍼
static std::vector<float3> g_objHostVel;
static std::vector<double> g_objRestRel;   // (x_rest − 자기 물체의 c0), N×3
static int    g_objRestN = 0;
static const float3* g_objRestSrc = nullptr;
// 물체 = 그래프 연결 성분 (여러 물체가 있는 장면에서 물체마다 따로 형상 유지·반발)
static int    g_objMinComponent = 50;       // 이보다 작은 성분(floater)은 물체로 보지 않는다 (id −1)
static int    g_objNumComponents = 0;       // 물체 수 K
static int    g_objNumSmall = 0;            // 무시한 작은 성분 수
static int    g_objSmallParticles = 0;
static int    g_objLargest[3] = { 0, 0, 0 };
static int    g_objCompN = 0;
static const int* g_objCompSrc = nullptr;   // d_nbrIdx 주소 (그래프 재구축 감지)
static int    g_objCompMinUsed = -1;
static std::vector<int>    g_objIdHost;     // 입자 → 물체 id (−1 = 물체 아님)
static std::vector<int>    g_objCount;      // 물체별 입자 수
static std::vector<float>  g_objC0;         // 물체별 rest 무게중심 [K×3]
static std::vector<double> g_objPrevVW;     // 물체별 직전 프레임 v, ω [K×6]
static std::vector<char>   g_objPrevOk;
static int*   d_objId = nullptr;            // [N]
static float* d_objParams = nullptr;        // [K×16] c0, c, R(행우선 9), valid
static float* d_objImpulse = nullptr;       // [K×10] c, Δv, Δω, fired
static int    g_objParamCap = 0;
// 물체 단위 shape matching 의 GPU 경로: 누적·극분해를 디바이스에서 한다 (위치 전체 다운로드·호스트 순회 없음).
// false = 기존 호스트 경로 (A/B 비교용으로 보존). 물체 수가 OBJ_GPU_MAX_K 를 넘으면 호스트 경로로 돈다.
#define OBJ_GPU_MAX_K 64
static bool    g_objShapeGPU = true;
static double* d_objRestRel = nullptr;      // [N×3] g_objRestRel 과 같은 값
static double* d_objSums = nullptr;         // [K×12] Σ(p−s), Σ(p−s)(x_rest−c0)ᵀ 행우선
static double* d_objShift = nullptr;        // [K×3] 누적 기준 s = 직전 프레임 무게중심 (double 누적의 상쇄 방지)
static double* d_objSumQ = nullptr;         // [K×3] Σ(x_rest − c0) — float c0 라 정확히 0 은 아니다
static float*  d_objC0Dev = nullptr;        // [K×3]
static int*    d_objCountDev = nullptr;     // [K]
// 붙잡힌 가우시안 (외부 물체 — 로봇 손가락 등): invMass 0 으로 고정하고 위치를 스텝마다 받는다 (setAttachedParticles).
// 붙잡는 동안 물체 단위 형상 유지의 강체 맞춤에 가우시안별 무게 d_objW 를 쓴다 — 붙잡힌 몇백 개가 몸 전체의 자세를 정하게.
// 무게 없이는 10만 개 중 몇백 개라 맞춤이 떨어지는 나머지를 따라가, 붙잡힌 곳만 남고 몸이 늘어나 빠진다.
static int*    d_attIdx = nullptr;
static float*  d_attPos = nullptr;
static int     g_attCap = 0;
static int     g_attCount = 0;
static bool    g_attActive = false;
static float*  d_invMassSaved = nullptr;    // [N] 붙잡기 전 invMass
static float*  d_objW = nullptr;            // [N] 강체 맞춤 무게 (붙잡힌 점 = weight, 나머지 1)
static float*  d_objDisp = nullptr;         // [K] 무게중심의 프레임 간 이동량
static int     g_objGpuCap = 0;
static int     g_objGpuRestCap = 0;
static bool    g_objShiftValid = false;     // d_objShift 가 직전 프레임 무게중심이다 (첫 프레임은 c0)
static double g_objHostMsAccum = 0.0;
static int    g_objTimedFrames = 0;
static int    g_objImpulseCount = 0;

// ── Self-collision: '처음 붙어 있던 쌍 제외' 규칙 ────────────────────────────
// rest 거리 < d_ex 인 쌍은 충돌시키지 않는다 (그래프가 이미 붙잡고 있고, 밀어내면 가만히 있어도 터진다).
// 떨어져 있던 부분(잎끼리, 안경 렌즈-볼, 머리카락 가닥)이 겹칠 때만 접촉이 생긴다.
// CPU 검증(scratchpad selfcol_experiment.py / selfcol_diag.py): rest 접촉 0,
// 연결 없는 두 시트 낙하 시 OFF는 100% 관통, ON은 A-B 3D 최소거리 p5 = 0.99·d_c (관통 없음, 반복 20=60 동일).
#define SELFCOL_MAX_CONTACTS 32
static bool   g_selfColEnabled = false;
// d_c = scale × 간격 (그래프 최소 엣지 길이의 중앙값). 1.5 근거(selfcol_radius.py, 시트 쌓기):
//   1.0 → 한 층이 다른 층 틈으로 끼어 수직 간격 0.26·d_c · 1.5 → 0.86·d_c (기하 기대 0.84) · 2.0 → 0.94·d_c
//   입자당 후보 최대 11 / 17 / 28 (상한 32), 셋 다 3D 관통 없음 (p5 ≥ 0.99·d_c)
static float  g_selfColRadiusScale = 1.5f;
static float  g_selfColExcludeScale = 2.0f;  // d_ex = scale × d_c
static float  g_selfColSearchScale = 1.5f;   // 탐색반경 = scale × d_c (predict 이후 반복 중 이동 여유)
static float  g_selfColSpacing = 0.0f;
static int    g_selfColSpacingN = 0;
static const float* g_selfColSpacingSrc = nullptr;
static int    g_selfColContacts = 0;         // 마지막 프레임 후보 수 (입자별 목록 합)
static int    g_selfColOverflow = 0;         // 목록 상한 초과로 먼 후보를 교체한 횟수
static double g_selfColBuildMsAccum = 0.0;
static long long g_selfColContactsAccum = 0;
static int    g_selfColTimedFrames = 0;
static int    g_selfColCap = 0;
static int    g_selfColTable = 0;
static int    g_selfColBits = 0;
static unsigned int* d_scKeys = nullptr;
static unsigned int* d_scKeysSorted = nullptr;
static int*   d_scIdx = nullptr;
static int*   d_scIdxSorted = nullptr;
static int*   d_scCellStart = nullptr;
static int*   d_scCellEnd = nullptr;
static int*   d_scContact = nullptr;
static int*   d_scCount = nullptr;
static float3* d_scDp = nullptr;
static char*  d_scSortTemp = nullptr;
static size_t g_scSortTempBytes = 0;
static float* d_scDisp2 = nullptr;           // [N] 이번 프레임 predict 변위² (최대값 → 격자 칸 크기)
static float* d_scMaxOut = nullptr;          // [1] cub 최대값 출력 (디바이스)
static char*  d_scReduceTemp = nullptr;
static size_t g_scReduceTempBytes = 0;
static float  g_selfColReach = 0.0f;         // 마지막 프레임 격자 칸 = 탐색반경 + 2·최대 변위 (상한 적용)
static int    g_selfColReachClamped = 0;     // 칸 크기가 상한에 걸린 프레임 수 (그보다 빠르면 관통 가능)
// 다른 물체 쌍 중 상대 이동이 fastRel·d_c 보다 큰 쌍만 평균 법선 규칙, 나머지는 거리 규칙.
// 0 = 다른 물체 쌍 전부 평균 법선 — CPU 복제에서 0.5(쉬는 접촉은 거리 규칙)가 바닥 위 층 간격을 더 빨리 무너뜨렸다
// (89프레임: 0 → 0.65·d_c, 0.5 → 0.05·d_c).
static float  g_selfColFastRel = 0.0f;
// 같은 물체 안 접촉(잎끼리·머리카락). 끄면 다른 물체 쌍만 보고, 다른 물체 경계상자 근처 입자만 격자에 넣는다.
// pillow(59만 점) 로그: 전체 입자를 넣으면 빠른 낙하 중 칸이 17.5·d_c 까지 커져 빌드 48~69 ms (12 FPS).
static bool   g_selfColWithinBody = false;
static int    g_selfColActive = 0;            // 마지막 프레임 활성 입자 수 (제한이 꺼져 있으면 N)
static unsigned char* d_scActive = nullptr;   // [N]
static float3* d_scDpCross = nullptr;         // [N] 해결 커널의 다른 물체 보정 성분 (반복 1회분)
static float3* d_scAccum = nullptr;           // [N] 프레임 누적 다른 물체 보정 (접촉 평균 강체 이동의 입력)
// 물체 경계상자 (objectShapeMatch 가 매 프레임 채운다)
static float* d_objBoxes = nullptr;           // [K×6] predict 위치 min(3), max(3)
static int    g_objBoxCap = 0;
static bool   g_objBoxesValid = false;
static float  g_objMaxBodyDisp = 0.0f;        // 프레임 간 물체 무게중심 이동량의 최댓값
static std::vector<double> g_objPrevCom;      // [K×3]
// 접촉 평균 강체 이동
static float* d_objPushSum = nullptr;         // [K×4] Σ 보정(3), 보정 받은 점 수
static float* d_objPushT = nullptr;           // [K×4] 이동량(3), valid
static int    g_objPushCap = 0;
static int    g_objPushFrames = 0;

// 바닥 접선 기저: n과 가장 덜 평행한 월드 축을 Gram-Schmidt. up이 축 정렬이면 체커가 월드 축과 나란하다.
// (GaussianView::updateGroundFrame과 같은 규칙이라 던지는 방향 0°가 체커 한 축과 일치한다)
static void groundTangentBasis(const float n[3], float t1[3], float t2[3])
{
	float a[3] = { 0.0f, 0.0f, 0.0f };
	const float ax = fabsf(n[0]), ay = fabsf(n[1]), az = fabsf(n[2]);
	if (ax <= ay && ax <= az) a[0] = 1.0f; else if (ay <= az) a[1] = 1.0f; else a[2] = 1.0f;
	const float d = a[0] * n[0] + a[1] * n[1] + a[2] * n[2];
	float x = a[0] - d * n[0], y = a[1] - d * n[1], z = a[2] - d * n[2];
	const float len = sqrtf(x * x + y * y + z * z);
	if (len > 1e-12f) { x /= len; y /= len; z /= len; }
	t1[0] = x; t1[1] = y; t1[2] = z;
	t2[0] = n[1] * z - n[2] * y;   // t2 = n × t1
	t2[1] = n[2] * x - n[0] * z;
	t2[2] = n[0] * y - n[1] * x;
}

// ── Volume Gaussian (부피 가우시안) ──────────────────────────
static bool g_useVolumeConstraint = false;
// Experimental A/B path: mesh-free Stable Neo-Hookean XPBD on the existing
// volume-cluster topology. It owns separate state and kernels so the legacy
// covariance-volume and shape-matching implementations remain untouched.
static bool  g_useGaussianNH = false;
static float g_gnhYoung = 1.0e4f;
static float g_gnhPoisson = 0.45f;
static float g_gnhComplianceScale = 1.0f;
// 거리 제약 Gauss-Seidel 경로 (A/B). 기존 Jacobi 경로는 그대로 두고, 켜면 거리 패스만
// 간선 컬러링 GS 로 바꾼다. 물체 규모(저주파) 변형의 수렴을 위한 것 — ablation/solver/gs_distance.py
static bool g_useDistanceGS = false;
// 부피 제약 Gauss-Seidel 경로 (A/B). 클러스터를 '멤버를 공유하지 않는 묶음'으로 칠해 색 순서로
// 제자리 갱신한다. 기본 OFF = 기존 결정론적 Jacobi gather. 색 수는 켜기 전에도 빌드 로그에 찍힌다.
static bool g_useVolumeGS = false;
// 솔버 반복 루프 시간 계측 (autorun 비교용, 기본 OFF — 켜면 매 substep 동기화가 한 번 생긴다)
static bool   g_physTiming = false;
static double g_physTimingSum = 0.0;
static int    g_physTimingCount = 0;
// Step 1에서는 alpha_vol을 손으로 튜닝한 상수로 둔다.
// (Step 5에서 alpha_vol = c_vol / (lambda_Lame * V_rest) 로 대체된다.)
static float g_volCompliance = 1e-6f;
// 고유값 이방성 임계. eig는 오름차순이므로 a1 = eig.x/eig.z, a2 = eig.y/eig.z.
// a1 > thr → volume, a2 > thr → surface, 그 외 → fiber.
static float g_volAnisoThreshold = 0.05f;
// 사전계산 결과 통계 (분류가 제대로 되는지 눈으로 보기 위한 것)
static int g_volMatCount[3] = { 0, 0, 0 };
static bool g_volPrecomputed = false;

// ── k-ring 볼륨 클러스터 ─────────────────────────────────────
// k=1: 그래프 직접 이웃 (기존과 동일). k>1: BFS k-hop 확장 클러스터.
// 응집 반경을 키워서 이웃 클러스터 간 그래디언트 방향을 정렬시킨다.
// (1-ring에서 "각자 부피는 지키는데 방향이 제각각 → 모래알 진동"이던 것을 해결)
static int g_volRingK = 1;
// 클러스터당 멤버 상한. BFS 결과가 이보다 크면 깊이 순서 균등 스트라이드로 서브샘플.
// 비용/메모리 상한이 k와 무관하게 고정된다. rest와 cur의 det를 '같은 멤버 리스트'로
// 재므로 서브샘플이어도 rest에서 J=1이 정확히 유지된다 (rest-J self-check가 검증).
static int g_volMaxMembers = 64;
// ── 반장(leader) 희소화: Poisson-disk ──
// leaderMinHop(r): 반장 간 그래프상 최소 간격. r=1이면 모든 노드가 반장(기존 동작).
// r≥2면 "이미 뽑힌 반장에서 r홉 이내는 반장이 안 된다" → 공간 균등 희소 배치.
// 반장이 아닌 노드는 클러스터를 만들지 않지만(계산량 감소), 여전히 이웃 반의 멤버로서
// 부피 보정을 받는다(자유도 N개 불변). r < k 여야 클러스터 반경이 커버 간격보다 커서
// 모든 노드가 최소 1개 반에 속함(커버리지 보장). 진단 ③에서 1/8까지 안전 확인됨.
static int g_volLeaderMinHop = 1;
// 반원 선발: 0 = BFS 비례 스트라이드(현재), 1 = farthest-point(방향 균등)
static int g_volMemberSelect = 1;
// ── h_j 상한 (양방향 유계 커버) ────────────────────────────────────────────
// 기존 greedy set cover는 '하한'만 보장했다(targetCover 이상). 상한이 없어서
// 인기 노드는 수천 개 클러스터의 멤버가 된다 — 실측 k=3에서 평균 64인데 최대 937.
// gather Kernel B는 스레드당 h_j만큼 도는데 워프는 최장 스레드를 기다리므로,
// 이 꼬리가 비용을 지배한다(k=2→6에서 작업량 1.2% 증가에 지연 3.3배).
// 상한을 걸면: 멤버 선발 시 이미 U만큼 덮인 후보를 건너뛴다.
//   → 커버리지가 [L, U]로 유계  →  GPU 부하 균등 + 과잉 중복 제거
// 0 = 비활성(기존 동작). 상한을 켜면 coverCnt를 순서대로 갱신해야 하므로
// r=1 경로도 병렬이 아닌 순차로 돈다(빌드 1회성 비용).
static int g_volCoverMax = 0;
static int g_volLeaderCount = 0; // 마지막 빌드의 실제 반장 수 M (통계)
static bool g_volTopoDirty = false; // k/cap 변경 시 다음 프레임에 리빌드
// 0 = scatter (atomicAdd 산란), 1 = gather (전치 CSR 읽기, atomic 없음/결정론적)
// 실측 결과 (k=6, squash 부하): J 거동 동일 (mean 0.9991 vs 0.9991),
// gather 19.5ms vs scatter 27.4ms (~30% 빠름) + 결정론적 → gather를 기본값으로.
static int g_volGatherMode = 1;
// Step 2 — J 시각화: SH 색 대신 J 컬러맵(파랑 J<1 / 흰색 1 / 빨강 J>1)으로 렌더
static bool g_volJVizEnabled = false;
static float g_volJVizGain = 3.0f; // (J-1)*gain 이 ±1에서 포화. 3이면 ±33% 변화가 최대 채도
// matType 시각화: 물질 분류가 실제 재질과 맞는지 눈으로 확인 (volume 회색 / surface 파랑 / fiber 빨강)
static bool g_volMatVizEnabled = false;

// ── 클램프 계측 ──────────────────────────────────────────
// 각 하드클램프가 "실제로 발동한" 횟수를 센다. 정석 솔버라면 정상 동작에서 0에 가까워야 한다.
// 자주 걸린다 = 그 지점이 불안정하다는 신호. (부피 경로만, 각도 제약은 미사용이라 제외)
//   [0]=J 클램프  [1]=스텝 클램프  [2]=Σ⁻¹ 고유값 하한(G4)
static bool g_volClampStats = true;
static long long g_volClampAccum[3] = { 0, 0, 0 };  // 프레임 누적
static long long g_volClampFrames = 0;

// ── Step 3: 물성 앵커 ──────────────────────────────────────
// 손 튜닝 상수 대신 물리 공식으로 클러스터별 compliance를 채운다:
//   α_vol(i) = c_vol / (λ_Lamé · V_rest(i)),   λ_Lamé = E·ν / ((1+ν)(1−2ν))
// c_vol = 1 이 이론값(사면체 세계의 공식 그대로). c_vol=1일 때의 J 잔차가
// "구조적 불일치의 크기"이고, FEM 캘리브레이션(Step 4~5)의 정량 근거가 된다.
// ※ E의 절대 스케일은 씬 단위가 미터가 아니라서 아직 명목값이다.
//   실제 단위 캘리브레이션은 FEM과 같은 좌표계를 쓰는 Step 4에서 확정된다.
static bool g_volUsePhysicalAlpha = false;
static float g_volMatE = 1.0e4f;  // Young's modulus (연조직 ~10 kPa 명목값)
static float g_volMatNu = 0.45f;  // Poisson ratio (0.5 = 완전 비압축, 0.49 상한)
static float g_volCvol = 1.0f;    // 보정 계수. 1 = 이론값

// ── 명찰/신분증에 가우시안 자신의 모양 반영 (mixture rest) ──
// V_rest와 matType 계산에 Σ_mix = Σ_centers + B 를 쓴다.
// B = 멤버 가우시안들의 렌더 공분산(R·S²·Rᵀ)의 opacity 가중 평균.
// ★ 저울(detRest ← 런타임 J의 분모)은 절대 건드리지 않는다 —
//   분자(detCur)가 중심점만으로 측정되므로 분모도 같은 경로여야 rest에서 J=1.
static bool g_volMixtureRest = false;

// ── Σ⁻¹ 축퇴 방어 방식 (A/B 토글) — 디바이스 상수 ──────────────────────────
// 목적은 하나 — 클러스터가 납작하면 Σ⁻¹이 폭발하므로 그걸 막는 것. 두 방식이 있다.
//
//   모드 0 (기존) : 고유분해 → 최소축을 최대축의 6%로 클램프 → V·D·Vᵀ 재조립 → 역행렬
//                   정밀하지만 반복 Jacobi라 ~600 FLOP.
//   모드 1 (신규) : Σ + eps·(trace/3)·I  → 역행렬.  ~15 FLOP.
//                   대각선에 같은 값을 더하면 세 고유값이 모두 올라가 역행렬이 유계가 된다.
//
// ★ 왜 중요한가: 이 구간은 멤버 수 n과 무관한 '직렬 구간'이다.
//   클러스터당 병렬화(워프/블록)를 하면 루프는 n/32로 줄지만 이건 안 줄어 Amdahl 병목이 된다.
//   n=257 기준: 병렬 13,621 FLOP vs 직렬 650 → 32스레드면 426+650 이라 직렬이 60%를 차지.
//
// ★ 모드 1은 '축퇴한 클러스터에만' 적용한다. Step 2에서 이미 계산한 detHat이 축퇴도를
//   나타내므로(등방이면 1, 납작할수록 0) 추가 비용 없이 판정된다. 건강한 클러스터는
//   Σ를 그대로 써서 모드 0의 V·D·Vᵀ 재조립 오차조차 없다.
// 기본값 = 1 (등방 정규화). 실측으로 물리 동일 확인:
//   부피가중 J  고유분해 0.99296  vs  등방 0.99302  (차이 0.006%p, k=3 cap=64 변위100%)
//   rest-J self-check 는 양쪽 1.00000
__device__ int   g_dSinvIsoReg = 1;       // 0 = 고유분해(기존), 1 = 등방 정규화
__device__ float g_dSinvIsoEps = 0.06f;   // 더할 양 = eps · (trace/3)
__device__ float g_dSinvIsoDetThr = 0.2f; // detHat이 이보다 작으면 축퇴로 보고 정규화
// ── 렌더 변형 F 추정: 공분산 쿠션(젤리) 실시간 토글 ─────────────────
//  0 = 기존(이웃 '중심점'만으로 PPt 구성)
//  1 = 이웃의 rest 공분산 Σ_j 를 PPt·QPt '양쪽'에 더함 → A 를 항등(변형없음)으로
//      당기는 정규화. 정보 없는 방향(평면 법선)만 A→I 로 채워 삐죽 방지, rest 에선 A=I 유지.
//      (PPt 에만 더하면 A 가 0쪽으로 눌려 모든 가우시안이 축소된다 — 초판 버그.)
//      λ 작을수록 관측 방향 편향(모양 감쇠) 작음. OFF면 기존과 완전 동일.
__device__ int   g_volCovReg    = 0;
__device__ float g_volCovLambda = 0.2f;
// ── 렌더 변형 F 추정: 축퇴 강건화 (2026-08-20) ────────────────────────
//  0 = 기존 경로, 1 = 강건 경로(기본). 증상: ficus 줄기·잎, lego, 안경이
//  "회전이 하나도 안 되고 이동만" + 삐죽. 원인 둘 —
//
//  ① det 임계값이 절대값이었다.  ok = inverse3x3_safe(PPt, ..., 1e-8f)
//     PPt 는 [길이²] 이라 det 는 [길이⁶]. 얼굴 실측(이웃거리 0.036, k=25)으로
//       사방으로 퍼진 이웃 : det ≈ 1.3e-6  → 통과
//       평면(잎)           : det ≈ 8.2e-10 → 실패
//       직선(줄기·안경)    : det ≈ 3.3e-13 → 실패
//     실패하면 preprocess 가 변형 경로를 통째로 건너뛰고 '원래 회전·스케일'을
//     그대로 쓴다(아래 `else if (... && ok)` 참조) → 회전이 멈춘다.
//     씬이 작으면 멀쩡한 이웃까지 같이 떨어진다(스케일 1/3 → det 1/729).
//     → 물리 커널의 sigmaScaleAndShapeDet 와 같이 trace 로 정규화해 '모양'만 본다.
//       이건 an earlier rendering path 과 같은 버그다(물리는 고쳤고 렌더는 안 고쳐져 있었다).
//
//  ② 정규화를 PPt(분모)에만 더했다. rest 에서도 F = PPt(PPt+εI)⁻¹ ≠ I 가 되어
//     고유값이 작은 방향이 λ/(λ+ε) 만큼 눌린다. ε = 1e-4·trace 기준
//       λ/trace = 0.33 → 0.9997 (무해)   1e-4 → 0.50   1e-6 → 0.0099 (100배 압착)
//     즉 얇은 잎의 법선, 가는 줄기의 단면이 뭉개진다.
//     → QPt 에도 같이 더한다. 그러면 rest 에서 F = (PPt+εI)(PPt+εI)⁻¹ = I 이고,
//       관측 불가 방향은 0 이 아니라 '변형 없음(I)' 으로 남는다.
//       수식으로는 항등행렬 쪽 Tikhonov: min Σ|q−Fp|² + ε|F−I|².
//       바로 아래 쿠션 코드는 이미 양쪽에 더하고 있었다 — 기본 경로만 빠져 있었다.
//
//  ①②는 세트다. ② 없이 ①만 하면 축퇴 방향으로 F 가 폭주하고,
//  ① 없이 ②만 하면 여전히 det 가 낮아 포기한다.
__device__ int g_renderFRobust = 1;
// 진단 카운터: [0]=변형 적용, [1]=det 실패(원본 유지), [2]=이웃<3(원본 유지)
// ⚠️ preprocess 는 frustum culling 이후라 '화면에 보이는' 가우시안만 센다.
__device__ unsigned int g_renderFCtr[3] = { 0, 0, 0 };
// ── [SH] 변형 회전을 겉모습(구면조화)에도 반영할 것인가 ────────────────
//  문제: 3DGS 의 방향성 색(SH)은 학습 시점의 '월드 좌표계'에 고정되어 있다.
//        변형으로 표면이 R 만큼 돌아도 SH 를 그대로 두면, 모양만 돌고
//        하이라이트는 월드 축에 붙박여 있다(회전이 큰 씬에서 눈에 띈다).
//  해법: 계수를 돌릴 필요가 없다. 방향을 Rᵀ 로 되돌려 평가하면 등가다.
//        f_new(ω) = f_old(R⁻¹ω) = f_old(Rᵀω)   ⟹ 3×3 곱 한 번
//  ⚠️ 여기 쓰는 R 은 아래 SVD 경로의 R 이 아니다. 그쪽 R 은 '클램프된' 특이값으로
//     나눠 만들어져 순수 회전이 아니다(A23). 조명은 R 을 단독으로 쓰므로
//     그 찌그러짐이 상쇄되지 않아 방향이 기운다. 그래서 polarRotationGLM 으로
//     따로 뽑는다 — 모양 경로는 건드리지 않으므로 기존 씬 결과가 불변이다.
//  0 = OFF(기존) · 1 = ON(Rᵀ, 정방향) · 2 = 진단용 역방향(R). 육안으로 1/2 비교 가능.
__device__ int g_shRotate = 0;
// [0]=회전 적용, [1]=폴백(축퇴/반전이라 항등으로 둠)
__device__ unsigned int g_shRotCtr[2] = { 0, 0 };
// ── [TN] 접선-법선 분해: 렌더 변형 F 의 '두 번째 경로' (기존 3D 최소제곱과 독립) ──
//  0 = OFF (기존 3D LS 그대로), 1 = ON
//  원리: 표면 splat 은 법선 방향 관측이 불가능(중심들이 한 평면에 깔림)해서 3D LS 가
//  ill-posed 였다. 그래서 관측 가능한 접선 2D 만 최소제곱으로 풀고(well-posed),
//  남은 법선 1자유도는 물리가 계산한 부피 J 로 닫는다: s_n = J / det(접선변형).
//  → 3×3 역행렬·고유분해가 2×2 역행렬 + 나눗셈 1회로 줄고, 임의 클램프가 필요없어진다.
__device__ int   g_tnMode = 0;
__device__ float g_tnFlatRatio = 0.15f;   // s_min/s_max 가 이보다 작아야 '납작'(= 이 경로 사용)
__device__ const float* g_dVolJPerG = nullptr; // [N] per-Gaussian 집계 J (전치 CSR 평균)
__device__ unsigned int g_tnUseCtr[2] = { 0, 0 }; // [0]=TN 적용, [1]=폴백(기존 경로)
// precompute가 쓸 렌더 속성 포인터 (FORWARD::preprocess 진입 시 저장됨)
static const glm::vec3* g_gsScalesPtr = nullptr;
static const glm::vec4* g_gsRotationsPtr = nullptr;
static void refreshPhysicalAlpha();
// k-ring 리빌드용 호스트 그래프 사본 (그래프는 로드 후 불변)
static std::vector<int> g_hGraphOffset, g_hGraphCount, g_hGraphIdx;

// ── 매 프레임 J 통계 (수치 검증용) ───────────────────────────
// 육안 검증만으로는 "제약이 작동하는가"를 판정할 수 없어서 별도로 수집한다.
// rest 상태에서 mean≈1, 눌리면 <1, 부풀면 >1. 볼륨 제약이 J를 1로 되돌리는지 본다.
static bool g_volCollectJStats = true;
static float* d_volJScratch = nullptr; // [N] 매 프레임 J 결과
static float* d_volJPerG = nullptr;    // [N] [TN] per-Gaussian 집계 J (전치 CSR 평균)
static int g_volJStatN = 0;
static float g_volJMean = 1.0f;
static float g_volJStd = 0.0f;
static float g_volJMin = 1.0f;
static float g_volJMax = 1.0f;
static float g_volJP05 = 1.0f;
static float g_volJP95 = 1.0f;
// ── 부피 가중 J — 전역 부피 오차의 정직한 지표 ────────────────────────────
// 위의 g_volJMean은 '클러스터 개수' 평균이라 좁쌀 클러스터와 큰 덩어리에 똑같이 1표를 준다.
// V_rest 스프레드가 실측 167만배(k=1)이므로 개수 평균은 작은 클러스터의 노이즈에 지배된다.
// 물리적으로 의미 있는 값은 부피로 가중한 것:
//     J_vw = Σ V_i·J_i / Σ V_i        (전역 부피비. 1이면 총부피 보존)
// the known limitation "전역 부피 보존 미보장"을 정량화하는 지표가 바로 이것.
//
// ★ 왜 '부피 가중'이 단순히 나은 평균이 아니라 정답 그 자체인가 (2줄 유도):
//     J_i = √(detΣ_cur/detΣ_rest) 이고 V = (4/3)π√(detΣ) 이므로
//     J_i = V_cur,i / V_rest,i     →     V_cur,i = J_i · V_rest,i
//     ⟹ Σ V_cur,i / Σ V_rest,i = Σ(J_i·V_rest,i)/Σ V_rest,i = J_vw   (정의상 동일)
//   개수 평균 (1/M)ΣJ_i 는 "클러스터를 무작위로 하나 집었을 때의 부피비 평균"이라
//   전역 부피 보존과는 아무 관계가 없는 양이다.
//
// ★★ 경고 — 가중치는 반드시 J와 '같은 종류'의 부피여야 한다.
//   J는 중심점 공분산의 비율이므로 가중치도 중심점 부피(mixture OFF)여야 위 상쇄가 성립한다.
//   d_V_rest는 g_volMixtureRest가 켜지면 det(Σ_c + B) 기반으로 바뀌므로,
//   그 상태에서는 J_vw가 "전체 물질 부피비"라는 다른 양이 된다(affine 가정 하에 이것도 유효한 등식).
//   둘은 서로 다른 숫자다 — 설정을 섞어 비교하지 말 것. 지금까지의 측정치는 전부 mixture OFF 기준.
static float g_volJVwMean = 1.0f;
static float g_volJVwStd = 0.0f;             // 부피 가중 표준편차
static std::vector<float> g_hVRestCache;     // [N] V_rest 호스트 캐시. 리빌드 때만 갱신
// [진단] 전치 CSR 호스트 캐시 — "나를 포함하는 클러스터" 목록. per-Gaussian J 집계 측정용.
static std::vector<int> g_hRevOffset, g_hRevCount, g_hRevIdx;
static bool g_jSmoothPending = false;        // UI 버튼 → 다음 J 계산 때 1회 측정
static char g_jSmoothTag[96] = "";
// [TN] 호스트 측 상태 (매 프레임 per-Gaussian J 갱신 여부를 결정)
static bool  g_tnModeHost = false;
static float g_tnFlatRatioHost = 0.15f;

// ── Squash Test (validation condition: 위 눌러서 옆으로 삐져나오나) ─
// applySeedCommands와 별개로, 프레임마다 재적용되는 슬랩 pin.
static bool g_squashActive = false;
static int g_squashAxis = 1;            // 0=X, 1=Y, 2=Z (씬마다 다름)
static float g_squashSlabPct = 0.08f;   // 위/아래 각 8%
static float g_squashRampPerSec = 0.15f;
static float g_squashMaxDisp = 0.30f;   // 축 방향 extent의 30%까지
static float g_squashSceneExtent = 1.0f;
static float g_squashCurDisp = 0.0f;
// 평판 경계조건. true = 미끄럼(하중축만 구속), false = no-slip(세 축 구속, 기존 경로).
// 기본값을 미끄럼으로 둔다 -- no-slip 은 FEM 참조해에 요소 반전을 일으켜 정답으로
// 쓸 수 없기 때문이다. A/B 를 위해 두 경로 모두 유지한다.
static bool g_squashSlip = true;
// Press 모드: 슬랩을 핀으로 잡지 않고, 위·아래 평판을 Ground 바닥과 같은 규칙(평판 밖으로 나간 중심만 평판 위로 투영,
// 마찰 없음)으로 건다. 바닥 충돌에서 부피 제약이 제대로 도는지 의심되어, 같은 변위·체크포인트에서 슬랩 핀과 A/B 하려는 것.
static bool  g_squashPress = false;       // 이번 실행의 모드 (Start 시 g_squashPressNext 에서 복사)
static bool  g_squashPressNext = false;   // UI 체크박스
static float g_squashLo = 0.0f;           // rest 축 방향 min (아래 평판)
static bool  g_squashPressFromLo = false;  // true = min 쪽 평판이 움직인다 (기본은 max 쪽). 좌표계의 '위'가 −축인 데이터용
static float g_squashHi = 0.0f;           // rest 축 방향 max (위 평판 = hi − curDisp)
// 실행이 새로 시작되면 metadata CSV 를 이어붙이지 않고 잘라낸다. 같은 라벨로 두 번
// 돌리면 좌표는 덮어써지는데 metadata 만 쌓여, 어느 설정이 어느 좌표를 만들었는지
// 알 수 없게 된다. 실제로 α 스윕 한 벌을 그렇게 잃었다.
static bool g_femMetaFresh = false;
static int* d_squashTopIdx = nullptr;
static int* d_squashBotIdx = nullptr;
static float3* d_squashTopRest = nullptr;
static float3* d_squashBotRest = nullptr;
static int g_squashTopCount = 0;
static int g_squashBotCount = 0;
// ── 고정 변위 자동 로깅 ────────────────────────────────────────────────────
// 손으로 버튼을 누르면 k마다 로그 시점의 변위가 달라져 비교가 성립하지 않는다
// (1차 k 스윕이 그래서 무효였다: k=1은 disp 1.23, k=2는 0.10에서 측정됨).
// 그래서 정해진 변위 비율에 도달하면 램프를 '멈추고' 솔버가 정착할 때까지 기다린 뒤
// 자동으로 J를 기록한다. 과도응답이 아닌 준정적 응답을 재는 것이 목적이다.
//   ※ 램프는 고정 dt로 진행되므로 프레임 수 = 변위. FPS가 달라도 하중 이력은 동일하다.
static const float g_squashCkpt[4] = { 0.25f, 0.50f, 0.75f, 1.00f };
static bool g_squashAutoLog = true;
static int  g_squashDwellFrames = 90;    // 체크포인트마다 정착시킬 프레임 수
static int  g_squashCkptIdx = 0;         // 다음에 걸릴 체크포인트
static int  g_squashDwell = 0;           // 남은 정착 프레임
static bool g_squashLogPending = false;  // 이번 프레임 J 통계가 나오면 기록

// ── FEM reference benchmark dump (opt-in, disabled by default) ─────────────
// The solver remains unchanged during normal rendering. When enabled, only
// the four settled squash checkpoints copy positions to the CPU and write CSV.
static bool g_femBenchmarkDumpEnabled = false;
static std::string g_femBenchmarkDumpDir = "./xpbd_dumps";
static std::string g_femBenchmarkRunLabel = "baseline";

static bool ensureFEMDirectory(const std::string& path)
{
	if (path.empty()) return false;
	std::string current;
	current.reserve(path.size());
	for (size_t i = 0; i < path.size(); ++i) {
		const char ch = path[i];
		current.push_back(ch);
		if (ch != '/' && ch != '\\' && i + 1 != path.size()) continue;
		if (current.size() <= 1) continue;
#ifdef _WIN32
		_mkdir(current.c_str());
#else
		mkdir(current.c_str(), 0755);
#endif
	}
	return true;
}

static std::string sanitizeFEMLabel(const std::string& label)
{
	std::string out;
	for (const char ch : label) {
		if (std::isalnum(static_cast<unsigned char>(ch)) || ch == '-' || ch == '_') out.push_back(ch);
		else if (ch == ' ' || ch == '.') out.push_back('_');
	}
	return out.empty() ? std::string("run") : out;
}

// Region Balloon: picked BFS region is treated as one covariance ellipsoid.
static bool g_useRegionBalloon = false;
static float g_regionBalloonCompliance = 1e-6f;
static float g_regionBalloonStrength = 0.35f;
static float g_regionBalloonMaxStep = 0.12f;
static int g_regionBalloonHops = 5;

static void setGpuCommandMode(bool enabled)
{
	g_gpuCommandMode = enabled;
	if (!enabled) {
		g_cmdIdx.clear();
		g_cmdDelta.clear();
	}
}

static void enqueueGpuCommand(int idx, const glm::vec3& delta)
{
	if (!g_gpuCommandMode) return;
	g_cmdIdx.push_back(idx);
	g_cmdDelta.push_back(delta);
}

static void drainGpuCommands(std::vector<int>& outIdx, std::vector<glm::vec3>& outDelta)
{
	outIdx.swap(g_cmdIdx);
	outDelta.swap(g_cmdDelta);
}

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped, const glm::mat3* R_sh = nullptr)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	// ★ [SH] 변형 회전 반영. 표면이 R 만큼 돌았으면 방향을 Rᵀ 로 되돌려 평가한다.
	//   (계수를 돌리는 것과 수학적으로 등가이고 훨씬 싸다 — 3×3 곱 한 번)
	//   rest 에서는 R = I 라 dir 이 그대로여야 한다. 색이 변하면 어딘가 틀린 것이다.
	if (R_sh != nullptr) {
		dir = (g_shRotate == 2) ? ((*R_sh) * dir) : (glm::transpose(*R_sh) * dir);
		const float dl = glm::length(dir);
		dir = (dl > 1e-20f) ? (dir / dl) : glm::vec3(0.0f, 0.0f, 1.0f);
	}

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix,
	
	float rotX, float rotY, float rotZ

	
	 )
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);


	//float c = cos(modRot);
	//float s = sin(modRot);



	float rx = glm::radians(rotX);
	float ry = glm::radians(rotY);
	float rz = glm::radians(rotZ);


	//float s = sin(angle_rad / 2.f);
	//float c = cos(angle_rad / 2.f);
	glm::mat3 Rx = glm::mat3(
		1, 0, 0,
		0, cos(rx), -sin(rx),
		0, sin(rx), cos(rx));

	glm::mat3 Ry = glm::mat3(
		cos(ry), 0, sin(ry),
		0, 1, 0,
		-sin(ry), 0, cos(ry));

	glm::mat3 Rz = glm::mat3(
		cos(rz), -sin(rz), 0,
		sin(rz), cos(rz), 0,
		0, 0, 1);

	// 최종 회전 조합 적용
	W = Rz * Ry * Rx * W;

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}
__device__ glm::vec4 createQuatFromAxisAngle(const glm::vec3& axis, float angle)
{
	float half_angle = angle * 0.5f;
	float s = sin(half_angle);
	float r = cos(half_angle);

	return glm::vec4(r, axis.x * s, axis.y * s, axis.z * s);  // (r, x, y, z)
}

__device__ inline void jacobiEigenDecomposition3x3(
	const float A_in[3][3],
	float eigenValues[3],
	float eigenVectors[3][3],
	const int maxIter = 50,
	const float eps = 1e-6f)
{
	// local copy (row-major)
	float A[3][3];
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			A[i][j] = A_in[i][j];

	// initialize eigenVectors to identity
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) eigenVectors[i][j] = (i == j) ? 1.0f : 0.0f;
	}
	float maxOff = 0.0f;
	int actual_iter = 0;
	for (int iter = 0; iter < maxIter; ++iter) {
		actual_iter = iter;
		// find largest off-diagonal |A[p][q]|
		int p = 0, q = 1;
		maxOff = fabsf(A[0][1]);
		if (fabsf(A[0][2]) > maxOff) { p = 0; q = 2; maxOff = fabsf(A[0][2]); }
		if (fabsf(A[1][2]) > maxOff) { p = 1; q = 2; maxOff = fabsf(A[1][2]); }

		if (maxOff < eps) break; // sufficiently diagonal

		// compute Jacobi rotation for indices (p,q) with p < q
		float app = A[p][p];
		float aqq = A[q][q];
		float apq = A[p][q];

		float phi = 0.5f * (aqq - app) / (apq + 1e-20f);
		// tan(2*theta) = 2*apq/(aqq-app)  but we use numerically stable formula
		float t = (phi >= 0.0f) ? (1.0f / (phi + sqrtf(1.0f + phi * phi))) : (1.0f / (phi - sqrtf(1.0f + phi * phi)));
		float c = 1.0f / sqrtf(1.0f + t * t);
		float s = t * c;

		// update A: only rows/cols p,q and off diag other elements
		float app_new = app - t * apq;
		float aqq_new = aqq + t * apq;

		A[p][p] = app_new;
		A[q][q] = aqq_new;
		A[p][q] = A[q][p] = 0.0f;

		// update the other entries
		for (int r = 0; r < 3; ++r) {
			if (r == p || r == q) continue;
			float arp = A[r][p];
			float arq = A[r][q];
			A[r][p] = A[p][r] = c * arp - s * arq;
			A[r][q] = A[q][r] = c * arq + s * arp;
		}

		// update eigenVectors: V = V * J  (apply rotation on columns p,q)
		for (int r = 0; r < 3; ++r) {
			float vrp = eigenVectors[r][p];
			float vrq = eigenVectors[r][q];
			eigenVectors[r][p] = c * vrp - s * vrq;
			eigenVectors[r][q] = s * vrp + c * vrq;
		}
	}
	// after convergence, diagonal entries are eigenvalues
	eigenValues[0] = A[0][0];
	eigenValues[1] = A[1][1];
	eigenValues[2] = A[2][2];

	// sort eigenvalues ascending and reorder eigenvectors accordingly
	// simple bubble sort for 3 elements
	for (int i = 0; i < 2; ++i) {
		for (int j = i + 1; j < 3; ++j) {
			if (eigenValues[i] > eigenValues[j]) {
				float tmp = eigenValues[i]; 
				eigenValues[i] = eigenValues[j]; 
				eigenValues[j] = tmp;
				// swap corresponding eigenvector columns
				for (int r = 0; r < 3; ++r) {
					float t = eigenVectors[r][i];
					eigenVectors[r][i] = eigenVectors[r][j];
					eigenVectors[r][j] = t;
				}
			}
		}
	}
}
__device__ inline void eigenDecomposition_glm(
	const glm::mat3& glmA, // 입력(대칭이라고 가정)
	glm::vec3& out_eigenValues,
	glm::mat3& out_eigenVectors) // columns are eigenvectors
{
	float A_row[3][3];
	// convert glm::mat3 (column-major) -> row-major A_row
	for (int r = 0; r < 3; ++r) {
		for (int c = 0; c < 3; ++c) {
			A_row[r][c] = glmA[c][r];
		}
	}

	float ev[3];
	float evecs[3][3];
	jacobiEigenDecomposition3x3(A_row, ev, evecs);

	// write back eigenvalues
	out_eigenValues = glm::vec3(ev[0], ev[1], ev[2]);

	// convert eigenvectors (evecs is row-major where column j is vector)
	// we stored eigenVectors[r][c] as row r, col c -> column c is eigenvector
	for (int c = 0; c < 3; ++c) {
		for (int r = 0; r < 3; ++r) {
			out_eigenVectors[c][r] = evecs[r][c]; // glm::mat3[col][row]
		}
	}
}


// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	glm::mat3 C_local(0.0f);

	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;
	

	// 2. 최종 스케일 = 체인메일 보정 행렬 * 원래 스케일

	float r = rot.x;
	float x = rot.y;
	float y = rot.z;
	float z = rot.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M =  S * R;
	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;//기존 코드rs strt

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}


__device__ void computeCov3D2(
	const bool bubble,
	const glm::vec3 scale,
	const glm::mat3& R_deform,
	const glm::mat3& S_deform,
	glm::mat3 A,
	float mod,
	const glm::vec4 rot,
	float* cov3D)
{
	glm::mat3 C_local(0.0f);




	// 1. 상대 제한: 원래 크기의 몇 배까지 허용할 것인가? (추천: 2.0 ~ 5.0)
	// 예: 3.0f면 원래 크기의 3배까지만 커질 수 있음.
	const float RELATIVE_GROWTH_LIMIT = 1.5f;
	const float MAX_ANISOTROPY = 0.7f;
	// 2. 절대 제한: 아무리 커져도 이 값(월드 좌표계)은 넘지 마라.
	// 사용자 데이터가 작다면 1.0f, 크다면 10.0f 등 조절 필요. (추천: 1.0f ~ 2.0f 부터 시작)
	const float ABSOLUTE_HARD_CAP = 0.15f;
	// [New] 표면 스무딩 파라미터
	// 1.0에 가까울수록 A를 그대로 사용, 0.0에 가까울수록 변형을 무시하고 원래 회전 유지
	// 값이 낮을수록 표면이 매끄러워지지만, 변형이 뻣뻣해질 수 있음. (추천: 0.7 ~ 0.9)
	const float DEFORMATION_STIFFNESS = 0.5f;

	// [New] 납작하게 누르기 (두께 압축)
	// 가장 짧은 축을 더 짧게 만들어서 "빈대떡"처럼 만듭니다.
	// 1.0이면 그대로, 0.1이면 10% 두께로 납작해짐. (추천: 0.3 ~ 0.6)
	const float FLATTENING_RATIO = 0.4f;
	const float MIN_THICKNESS_RATIO = 0.4f;
	const float MAX_ASPECT_RATIO = 2.9f;    // 장축이 단축의 3배를 넘지 못하게 함 (뭉뚝하게)
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;


	//// 2. 최종 스케일 = 체인메일 보정 행렬 * 원래 스케일

	float r = rot.x;
	float x = rot.y;
	float y = rot.z;
	float z = rot.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);
	//glm::mat3 M = S * R;
	//glm::mat3 M = S * R * S_deform * R_deform;
	glm::mat3 M = S * R * S_deform * glm::transpose(R_deform);// Ra Sa (R S S^T R^T) Sa^T Ra^T
	//glm::mat3 M = S * R  * glm::transpose(A);//Sa Ra (R S S^T R^T) Ra^T Sa^T = A sigma A^T

	
	float max_orig_scale = fmaxf(scale.x, fmaxf(scale.y, scale.z));

	//// 2. 허용 가능한 최대 반지름 결정
	////    (원래크기 * mod * 3배) 와 (절대값 1.0) 중 더 작은 것을 선택
	////    이렇게 하면 작은 가우시안은 작게 유지되고, 원래 큰 건 좀 더 커질 수 있음.
	float allowed_radius = max_orig_scale * mod * 50.0f;

	// 절대 제한(ABSOLUTE_HARD_CAP)을 넘지 못하게 막음 (안전장치)
	if (allowed_radius > ABSOLUTE_HARD_CAP) {
		allowed_radius = ABSOLUTE_HARD_CAP;
	}
	bool is_invalid = false;
	//// 각 축을 검사하여 제한
	//for (int i = 0; i < 3; ++i) {
	//	// NaN / Inf 체크 (시뮬레이션 폭발 감지)
	//	//1000배까지만 커진다면 그냥 100에서 바로 잘라도됨
	//	float current_len = glm::length(M[i]);
	//	if (isnan(M[i].x) || isnan(M[i].y) || isnan(M[i].z) ||
	//		isinf(M[i].x) || isinf(M[i].y) || isinf(M[i].z)) {
	//		float s = allowed_radius / (current_len + 1e-7f);
	//		M[i] *= s;
	//		//M = glm::mat3(0.0f);
	//		//break;
	//	}
	//	// 허용된 반지름보다 크면 강제로 줄임
	//	if (current_len > allowed_radius&& !bubble) {
	//		// 하드하게 s를 곱하는 대신, 초과분의 증가 속도를 늦춥니다.
	//		float overflow = current_len / allowed_radius;
	//		// 멱함수나 로그를 사용하여 부드럽게 억제 (예: overflow의 0.2승만 반영)
	//		float soft_s = (allowed_radius * powf(overflow, 0.2f)) / current_len;
	//		M[i] *= soft_s;
	//		//float s = allowed_radius / (current_len + 1e-7f);
	//		//M[i] *= s;			
	//		//break;
	//	}
	//}
		// Compute 3D world covariance matrix Sigma
		//glm::mat3 Sigma =  M * glm::transpose(M);
		glm::mat3 Sigma_orig = transpose(M) * M;  // R^T S^2 R
		//// 변형 텐서 F = R_deform * S_deform (극분해 결과 재결합)
		//glm::mat3 F = R_deform * S_deform; // 두 개 모두 사용
		//
		//// 논문 Eq.6 직접 적용
		//glm::mat3 Sigma = F * Sigma_orig * glm::transpose(F);
		// Covariance is symmetric, only store upper right
		cov3D[0] = Sigma_orig[0][0];
		cov3D[1] = Sigma_orig[0][1];
		cov3D[2] = Sigma_orig[0][2];
		cov3D[3] = Sigma_orig[1][1];
		cov3D[4] = Sigma_orig[1][2];
		cov3D[5] = Sigma_orig[2][2];
}


// ═══════════════════════════════════════════════════════════════════════════
// [TN] 접선-법선 분해로 렌더 공분산 계산 — 기존 3D 최소제곱과 완전히 독립된 경로
// ---------------------------------------------------------------------------
//  기존:  이웃 중심점 3D 변위 → 3×3 최소제곱 F  →  법선 방향 정보가 없어 ill-posed
//         → invPPt 폭주 → 가우시안 삐죽 → [1e-3, 3.0] 하드 클램프로 응급처치
//  신규:  ① 이 가우시안의 주축(t1,t2,n) — Σ=R diag(s²)Rᵀ 라 쿼터니언 R 의 열이 곧 주축.
//            고유분해 불필요. 최소 scale 축 = 표면 법선.
//         ② 접선 2D 최소제곱만 푼다 (2×2 역행렬, 표면에서 well-conditioned)
//            → 변형된 접선 a1, a2
//         ③ 법선 방향 n' = normalize(a1×a2)  (굽힘 시 법선이 함께 회전 = coupling 처리)
//            법선 '두께' s_n = J / |a1×a2|    ← ★ 관측 못 하는 1자유도를 물리 J 가 닫는다
//         ④ F = a1 t1ᵀ + a2 t2ᵀ + s_n n' nᵀ  →  Σ' = F Σ₀ Fᵀ
//  실패(납작하지 않음/접선 축퇴/J 없음)하면 false 를 반환해 기존 경로로 폴백한다.
// ═══════════════════════════════════════════════════════════════════════════
__device__ inline bool computeTangentNormalCov(
	int idx, int P,
	const float* realOrigin_points,   // rest 위치
	const float* orig_points,         // 현재(물리 후) 위치
	const int* nbr_idx, int nbr_off, int kn,
	const glm::vec3 scale, const glm::vec4 rot, float mod,
	float* out_cov6)
{
	// ── ① 주축: Σ = R diag(s²) Rᵀ 이므로 R 의 열이 고유벡터, s² 이 고유값 ──
	const float r = rot.x, x = rot.y, y = rot.z, z = rot.w;
	const glm::mat3 Rg = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y));
	// Rg 는 수학적 R 의 전치(glm 열우선)라, 주축 j = Rg 의 '행' j
	glm::vec3 ax[3];
	ax[0] = glm::vec3(Rg[0][0], Rg[1][0], Rg[2][0]);
	ax[1] = glm::vec3(Rg[0][1], Rg[1][1], Rg[2][1]);
	ax[2] = glm::vec3(Rg[0][2], Rg[1][2], Rg[2][2]);
	const float sv[3] = { fabsf(scale.x), fabsf(scale.y), fabsf(scale.z) };

	int jn = 0;                                   // 최소축 = 법선
	if (sv[1] < sv[jn]) jn = 1;
	if (sv[2] < sv[jn]) jn = 2;
	const float smax = fmaxf(sv[0], fmaxf(sv[1], sv[2]));
	if (smax <= 1e-20f) return false;
	// 충분히 납작할 때만 이 경로 (두꺼운 3D 가우시안은 기존 3D LS 가 옳다)
	if (sv[jn] / smax > g_tnFlatRatio) return false;
	const glm::vec3 t1 = ax[(jn + 1) % 3];
	const glm::vec3 t2 = ax[(jn + 2) % 3];
	const glm::vec3 nn = ax[jn];

	// ── ② 접선 2D 최소제곱: rest 접선좌표(u,v) → 현재 3D 변위 ──
	const glm::vec3 xi0(realOrigin_points[3 * idx], realOrigin_points[3 * idx + 1], realOrigin_points[3 * idx + 2]);
	const glm::vec3 xi(orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2]);
	float p00 = 0.f, p01 = 0.f, p11 = 0.f;
	glm::vec3 Qc0(0.f), Qc1(0.f);
	int used = 0;
	for (int k = 0; k < kn; ++k) {
		const int nj = nbr_idx[nbr_off + k];
		if (nj < 0 || nj >= P) continue;
		const glm::vec3 pj(realOrigin_points[3 * nj], realOrigin_points[3 * nj + 1], realOrigin_points[3 * nj + 2]);
		const glm::vec3 qj(orig_points[3 * nj], orig_points[3 * nj + 1], orig_points[3 * nj + 2]);
		const glm::vec3 p = pj - xi0;
		const glm::vec3 q = qj - xi;
		const float u = glm::dot(p, t1), v = glm::dot(p, t2);
		p00 += u * u; p01 += u * v; p11 += v * v;
		Qc0 += q * u;  Qc1 += q * v;
		++used;
	}
	if (used < 3) return false;

	// 2×2 역행렬 (3×3 과 달리 표면에서 조건수가 좋다 — 이게 이 방법의 핵심 이득)
	const float det2 = p00 * p11 - p01 * p01;
	const float trc = p00 + p11;
	if (!(det2 > 1e-12f * trc * trc)) return false;   // 접선조차 축퇴(선형 배열) → 폴백
	const float i00 = p11 / det2, i01 = -p01 / det2, i11 = p00 / det2;
	const glm::vec3 a1 = Qc0 * i00 + Qc1 * i01;       // 변형된 t1
	const glm::vec3 a2 = Qc0 * i01 + Qc1 * i11;       // 변형된 t2

	// ── ③ 법선: 방향은 접선 외적, 두께는 물리 J 가 결정 ──
	glm::vec3 cr = glm::cross(a1, a2);
	const float area = glm::length(cr);               // = |det(접선 변형)|
	if (!(area > 1e-20f)) return false;
	glm::vec3 np = cr / area;
	if (glm::dot(np, nn) < 0.0f) np = -np;            // rest 법선과 부호 일관 (뒤집힘 방지)

	float J = 1.0f;
	if (g_dVolJPerG != nullptr) {
		const float jv = g_dVolJPerG[idx];
		if (isfinite(jv) && jv > 1e-6f) J = jv;
	}
	// ★ 관측 불가능한 유일한 자유도를 물리가 닫는다
	float sn = J / area;
	sn = fminf(4.0f, fmaxf(0.25f, sn));               // 이상치 안전캡 (실측 p95 2% → 거의 미발동)

	// ── ④ F 조립 → Σ' = F Σ₀ Fᵀ ──
	const glm::mat3 F = glm::outerProduct(a1, t1) + glm::outerProduct(a2, t2)
		+ glm::outerProduct(np * sn, nn);

	glm::mat3 S3(0.f);
	S3[0][0] = mod * scale.x; S3[1][1] = mod * scale.y; S3[2][2] = mod * scale.z;
	const glm::mat3 M0 = S3 * Rg;
	const glm::mat3 Sig0 = glm::transpose(M0) * M0;   // 기존 computeCov3D 와 동일 정의
	const glm::mat3 Sig = F * Sig0 * glm::transpose(F);

	if (!isfinite(Sig[0][0]) || !isfinite(Sig[1][1]) || !isfinite(Sig[2][2])) return false;
	out_cov6[0] = Sig[0][0]; out_cov6[1] = Sig[0][1]; out_cov6[2] = Sig[0][2];
	out_cov6[3] = Sig[1][1]; out_cov6[4] = Sig[1][2]; out_cov6[5] = Sig[2][2];
	return true;
}





// CUDA 코드 상단(전역 영역)에 추가
//namespace FORWARD {
//	__constant__ float mytime[1];
//}
//// 렌더링 루프 내부 또는 시간 업데이트 지점에서
//float current_time = get_current_time(); // 사용자 정의 시간 획득 함수
//cudaMemcpyToSymbol(mytime, &current_time, sizeof(float));
// Perform initial steps for each Gaussian prior to rasterization.


__device__ float3 expand_contract(float3 pos, float t, float3 center, float scale)
{
	float3 dis = make_float3( pos.x - center.x, pos.y - center.y, pos.z - center.z);
	float len = sqrtf(dis.x * dis.x + dis.y * dis.y + dis.z * dis.z);
	float size = sinf(t * 0.5f) * scale*1.3f; // scale: 전체 볼륨 크기의 10% 등
	float s = (len > 1e-6f) ? ((len + size) / len) : 1.0f;
	dis.x *= s;
	dis.y *= s;
	dis.z *= s;
	float3 retval = make_float3(center.x + dis.x, center.y + dis.y, center.z + dis.z);	
	// floorf는 계단현상이 필요할 때만 사용
	// retval.x = floorf(retval.x); retval.y = floorf(retval.y); retval.z = floorf(retval.z);
	return retval;
}
__device__ float3 twist(float3 pos, float t, float3 center, float theta_scale, float& out_theta)
{
	float3 dis = make_float3(pos.x - center.x, pos.y - center.y, pos.z - center.z);
	float3 dis_xy = make_float3(dis.x, dis.y, 0.0f);
	float theta = sinf(t * 0.3f) * (pos.z - center.z) * theta_scale;
	out_theta = theta; // out parameter로 회전 각도 전달

	float c = cosf(theta);
	float s = sinf(theta);
	float dx = c * dis_xy.x - s * dis_xy.y;
	float dy = s * dis_xy.x + c * dis_xy.y;
	float3 retval;
	retval.x = center.x + dx;
	retval.y = center.y + dy;
	retval.z = pos.z;
	return retval;
}
//__device__ float3 twist(float3 pos, float t, float3 center, float theta_scale, float& out_theta)
//{
//	// 중심 기준으로 좌표 변환
//	float3 dis = make_float3(pos.x - center.x, pos.y - center.y, pos.z - center.z);
//
	// (x, z) 평면에서 y축을 중심으로 회전
//	float theta = sinf(t * 0.3f) * (pos.y - center.y) * theta_scale;
//	out_theta = theta; // out parameter로 회전 각도 전달
//
//	float c = cosf(theta);
//	float s = sinf(theta);
//	float dx = c * dis.x + s * dis.z;
//	float dz = -s * dis.x + c * dis.z;
//
//	float3 retval;
//	retval.x = center.x + dx;
//	retval.y = pos.y; // y는 그대로
//	retval.z = center.z + dz;
//	return retval;
//}
__device__ float3 wave(float3 pos,float t)
{
	// Wave 변형 파라미터 (원하는 대로 조정)
	float amplitude = 0.2f;  // 파동 높이
	float freq = 2.0f;       // 파동 주파수
	float speed = 3.0f;      // 시간에 따른 속도

	// 예시: Y축을 sin 파동으로 변형
	float wave = amplitude * sinf(freq * pos.x + speed * t);
	float3 retval;
	// X, Y, Z 각각에 웨이브를 다르게 줄 수도 있음
	float3 p_wave;
	retval.x = pos.x + wave; 
	retval.y = pos.y + amplitude * cosf(freq * pos.x + speed * t);
	retval.z = pos.z + amplitude * sinf(freq * pos.z + speed * t);

	return retval;
	// 결과 저장
}




// ===== math utils (device/host 공용) =====
__host__ __device__ inline float3 make_f3(float x, float y, float z) { return make_float3(x, y, z); }
__host__ __device__ inline float3 load_f3(const float* a, int i) { return make_float3(a[3 * i + 0], a[3 * i + 1], a[3 * i + 2]); }

// ===== 커널 상단 (또는 별도 헤더)에 유틸 =====
struct Mat3 {
	float m[3][3];
	__device__ static Mat3 I() { Mat3 A{}; A.m[0][0] = A.m[1][1] = A.m[2][2] = 1.f; return A; }
};

__device__ inline float3 operator+(const float3& a, const float3& b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ inline float3 operator-(const float3& a, const float3& b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ inline float3 operator*(const float s, const float3& a) { return make_float3(s * a.x, s * a.y, s * a.z); }
__device__ inline float  dot3(const float3& a, const float3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

__device__ inline Mat3 outer(const float3& a, const float3& b) {
	Mat3 M{};
	M.m[0][0] = a.x * b.x; M.m[0][1] = a.x * b.y; M.m[0][2] = a.x * b.z;
	M.m[1][0] = a.y * b.x; M.m[1][1] = a.y * b.y; M.m[1][2] = a.y * b.z;
	M.m[2][0] = a.z * b.x; M.m[2][1] = a.z * b.y; M.m[2][2] = a.z * b.z;
	return M;
}
__device__ inline Mat3 add(const Mat3& A, const Mat3& B) {
	Mat3 C{};
#pragma unroll
	for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) C.m[r][c] = A.m[r][c] + B.m[r][c];
	return C;
}
__device__ inline Mat3 mul(const Mat3& A, const Mat3& B) {
	Mat3 C{};
#pragma unroll
	for (int r = 0; r < 3; ++r) {
		for (int c = 0; c < 3; ++c) {
			C.m[r][c] = A.m[r][0] * B.m[0][c] + A.m[r][1] * B.m[1][c] + A.m[r][2] * B.m[2][c];
		}
	}
	return C;
}
__device__ inline Mat3 trans(const Mat3& A) {
	Mat3 B{};
#pragma unroll
	for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) B.m[r][c] = A.m[c][r];
	return B;
}
__device__ inline float3 mul(const Mat3& A, const float3& v) {
	return make_float3(
		A.m[0][0] * v.x + A.m[0][1] * v.y + A.m[0][2] * v.z,
		A.m[1][0] * v.x + A.m[1][1] * v.y + A.m[1][2] * v.z,
		A.m[2][0] * v.x + A.m[2][1] * v.y + A.m[2][2] * v.z
	);
}

// 3x3 대칭행렬(B=F^T F)의 역제곱근 근사: 뉴턴?슐츠 2회(빠르고 충분히 안정적)
__device__ inline Mat3 inv_sqrt_sym(const Mat3& B) {
	// normalize for stability
	float t = B.m[0][0] + B.m[1][1] + B.m[2][2];
	float s = fmaxf(t / 3.f, 1e-6f);
	Mat3 A{};
	for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) A.m[r][c] = B.m[r][c] / s;

	Mat3 Y = A;                   // target
	Mat3 X = Mat3::I();           // approx to A^{-1/2}
	const float alpha = 1.5f;
	// two Newton-Schulz iterations
#pragma unroll
	for (int it = 0; it < 2; ++it) {
		Mat3 XYA = mul(X, Y);
		Mat3 AX = mul(Y, X);
		Mat3 M{};
		for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c)
			M.m[r][c] = 0.5f * (3.f * Mat3::I().m[r][c] - (XYA.m[r][c] + AX.m[r][c]) * 0.5f);
		X = mul(X, M);
		Y = mul(M, Y);
	}
	// scale back
	for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) X.m[r][c] /= sqrtf(s);
	return X; // ? B^{-1/2}
}

__device__ inline glm::vec4 rotmat_to_quat(const Mat3& R) {
	float trace = R.m[0][0] + R.m[1][1] + R.m[2][2];
	glm::vec4 q;
	if (trace > 0.f) {
		float s = sqrtf(trace + 1.f) * 2.f; // 4w
		q.w = 0.25f * s;
		q.x = (R.m[2][1] - R.m[1][2]) / s;
		q.y = (R.m[0][2] - R.m[2][0]) / s;
		q.z = (R.m[1][0] - R.m[0][1]) / s;
	}
	else {
		int i = (R.m[0][0] < R.m[1][1]) ? ((R.m[1][1] < R.m[2][2]) ? 2 : 1) : ((R.m[0][0] < R.m[2][2]) ? 2 : 0);
		float a = R.m[i][i];
		int j = (i + 1) % 3, k = (i + 2) % 3;
		float s = sqrtf(1.f + a - R.m[j][j] - R.m[k][k]) * 2.f;
		float qarr[4] = { 0,0,0,0 };
		qarr[i] = 0.25f * s;
		q.w = (R.m[k][j] - R.m[j][k]) / s;
		qarr[j] = (R.m[j][i] + R.m[i][j]) / s;
		qarr[k] = (R.m[k][i] + R.m[i][k]) / s;
		q.x = qarr[0]; q.y = qarr[1]; q.z = qarr[2];
	}
	// normalize
	float l = sqrtf(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w) + 1e-8f;
	q.x /= l; q.y /= l; q.z /= l; q.w /= l;
	return q;
}
__device__ inline float length3(const float3& v) {
	return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

// ===== 3x3 SVD (Kabsch용) 보조 루틴 =====
__device__ void jacobiEigenSym3(const glm::mat3& A, glm::mat3& V, glm::vec3& eval) {
	glm::mat3 D = A;
	V = glm::mat3(1.0f);

	for (int it = 0; it < 10; ++it) {
		// 최대 절대 비대각 원소 선택
		int p = 0, q = 1;
		float a01 = fabsf(D[1][0]), a02 = fabsf(D[2][0]), a12 = fabsf(D[2][1]);
		if (a02 > a01 && a02 > a12) { p = 0; q = 2; }
		else if (a12 > a01) { p = 1; q = 2; }

		if (fabsf(D[q][p]) < 1e-10f) break;

		float app = D[p][p], aqq = D[q][q], apq = D[q][p];
		float phi = 0.5f * atanf(2.0f * apq / (aqq - app + 1e-20f));
		float c = cosf(phi), s = sinf(phi);

		// 회전 적용
		for (int k = 0; k < 3; ++k) {
			float dkp = D[p][k], dkq = D[q][k];
			D[p][k] = c * dkp - s * dkq;
			D[q][k] = s * dkp + c * dkq;

			float vkp = V[p][k], vkq = V[q][k];
			V[p][k] = c * vkp - s * vkq;
			V[q][k] = s * vkp + c * vkq;
		}
	}

	eval.x = D[0][0]; eval.y = D[1][1]; eval.z = D[2][2];
}

__device__ void svd3x3(const glm::mat3& M, glm::mat3& U, glm::vec3& S, glm::mat3& Vt) {
	glm::mat3 MtM = glm::transpose(M) * M;
	glm::mat3 V; glm::vec3 eval;
	jacobiEigenSym3(MtM, V, eval);

	// 고유값 내림차순 정렬
	int order[3] = { 0,1,2 };
	auto swapi = [&](int a, int b) { int t = order[a]; order[a] = order[b]; order[b] = t; };
	if (eval[order[0]] < eval[order[1]]) swapi(0, 1);
	if (eval[order[0]] < eval[order[2]]) swapi(0, 2);
	if (eval[order[1]] < eval[order[2]]) swapi(1, 2);

	glm::mat3 Vsorted;
	glm::vec3 Sdiag;
	for (int i = 0; i < 3; ++i) {
		int oi = order[i];
		Vsorted[i] = V[oi]; // 열 복사
		Sdiag[i] = sqrtf(fmaxf(eval[oi], 0.0f));
	}

	glm::mat3 VinvSigma(0.0f);
	for (int i = 0; i < 3; ++i) {
		if (Sdiag[i] > 1e-8f) VinvSigma[i][i] = 1.0f / Sdiag[i];
	}

	glm::mat3 Utmp = M * Vsorted * VinvSigma;

	// Gram-Schmidt 정규화
	for (int i = 0; i < 3; ++i) {
		glm::vec3 col(Utmp[i][0], Utmp[i][1], Utmp[i][2]);
		float n = glm::length(col);
		if (n < 1e-8f) { col = glm::vec3(0.0f); col[i] = 1.0f; n = 1.0f; }
		col /= n;
		for (int j = 0; j < 3; ++j) Utmp[i][j] = col[j];
	}

	U = Utmp;
	Vt = glm::transpose(Vsorted);
	S = Sdiag;
}

__device__ inline glm::mat3 quatToMat3(const glm::vec4& rot) {
	float r = rot.x, x = rot.y, y = rot.z, z = rot.w;
	return glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);
}

// 3x3 행렬의 역행렬을 계산하는 __device__ 함수
__device__ __inline__ glm::mat3 inverse(const glm::mat3& m) {
	// 행렬식(determinant) 계산
	float det = m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2]) -
		m[1][0] * (m[0][1] * m[2][2] - m[2][1] * m[0][2]) +
		m[2][0] * (m[0][1] * m[1][2] - m[1][1] * m[0][2]);

	// 행렬식이 0에 가까우면 (특이 행렬), 단위 행렬 반환
	if (abs(det) < 1e-8f) {
		return glm::mat3(1.0f);
	}

	float inv_det = 1.0f / det;
	glm::mat3 inv;

	// 수반 행렬(Adjugate Matrix)을 사용하여 역행렬 계산
	// GLM은 Column-Major 순서이므로 inv[col][row] 형태로 대입합니다.
	inv[0][0] = (m[1][1] * m[2][2] - m[2][1] * m[1][2]) * inv_det;
	inv[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det;
	inv[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det;
	inv[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det;
	inv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
	inv[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det;
	inv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;
	inv[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det;
	inv[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det;

	return inv;
}


// 공분산 갱신 (대칭 3x3 → 6개 요소)

__device__ inline glm::mat3 quat_to_mat3(const glm::vec4& q) {
	float x = q.x, y = q.y, z = q.z, w = q.w;
	float x2 = x * x, y2 = y * y, z2 = z * z;
	float xy = x * y, xz = x * z, yz = y * z;
	float wx = w * x, wy = w * y, wz = w * z;
	return glm::mat3(1 - 2 * (y2 + z2), 2 * (xy - wz), 2 * (xz + wy),
		2 * (xy + wz), 1 - 2 * (x2 + z2), 2 * (yz - wx),
		2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (x2 + y2));
}

// 회전 행렬을 쿼터니언으로 변환하는 수치적으로 안정적인 __device__ 함수
__device__ __inline__ glm::vec4 mat3_to_quat(const glm::mat3& m) {
	float t;
	glm::vec4 q;

	// 행렬의 대각합(trace)을 확인하여 가장 안정적인 계산법 선택
	float trace = m[0][0] + m[1][1] + m[2][2];

	if (trace > 0.0f) {
		t = sqrtf(trace + 1.0f) * 2.0f;
		q.w = 0.25f * t;
		q.x = (m[2][1] - m[1][2]) / t;
		q.y = (m[0][2] - m[2][0]) / t;
		q.z = (m[1][0] - m[0][1]) / t;
	}
	else if ((m[0][0] > m[1][1]) && (m[0][0] > m[2][2])) {
		t = sqrtf(m[0][0] - m[1][1] - m[2][2] + 1.0f) * 2.0f;
		q.x = 0.25f * t;
		q.y = (m[1][0] + m[0][1]) / t;
		q.z = (m[0][2] + m[2][0]) / t;
		q.w = (m[2][1] - m[1][2]) / t;
	}
	else if (m[1][1] > m[2][2]) {
		t = sqrtf(m[1][1] - m[0][0] - m[2][2] + 1.0f) * 2.0f;
		q.y = 0.25f * t;
		q.x = (m[1][0] + m[0][1]) / t;
		q.z = (m[2][1] + m[1][2]) / t;
		q.w = (m[0][2] - m[2][0]) / t;
	}
	else {
		t = sqrtf(m[2][2] - m[0][0] - m[1][1] + 1.0f) * 2.0f;
		q.z = 0.25f * t;
		q.x = (m[0][2] + m[2][0]) / t;
		q.y = (m[2][1] + m[1][2]) / t;
		q.w = (m[1][0] - m[0][1]) / t;
	}

	// 쿼터니언 정규화 (필요시)
	// return glm::normalize(q);
	return q;
}

// 쿼터니언 곱셈을 위한 __device__ 함수
__device__ __inline__ glm::vec4 quat_multiply(const glm::vec4& q1, const glm::vec4& q2) {
	glm::vec4 result;
	result.w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z;
	result.x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y;
	result.y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x;
	result.z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w;
	return result;
}
// glm::mat3를 Eigen::Matrix3f로 변환하는 함수
Eigen::Matrix3f glmToEigen(const glm::mat3& m) {
	Eigen::Matrix3f em;
	// glm은 column-major, Eigen도 기본적으로 column-major이므로 바로 복사
	memcpy(em.data(), &m[0][0], 9 * sizeof(float));
	return em;
}

// Eigen::Matrix3f를 glm::mat3로 변환하는 함수
glm::mat3 eigenToGlm(const Eigen::Matrix3f& em) {
	glm::mat3 m;
	memcpy(&m[0][0], em.data(), 9 * sizeof(float));
	return m;
}

// Eigen::Vector3f를 glm::vec3로 변환하는 함수
glm::vec3 eigenToGlm(const Eigen::Vector3f& ev) {
	return glm::vec3(ev.x(), ev.y(), ev.z());
}


/**
 * @brief glm::mat3 행렬의 SVD를 계산합니다.
 * @param F 입력 행렬 (3x3)
 * @param U 출력 행렬 U (3x3)
 * @param Sigma 출력 특이값 벡터 (3x1)
 * @param V 출력 행렬 V (3x3)
 */
__device__ inline void svd(const glm::mat3& F, glm::mat3& U, glm::vec3& Sigma, glm::mat3& V)
{
	// Transpose F, as the algorithm is formulated for row-major matrices
	glm::mat3 Ft = glm::transpose(F);
	glm::mat3 Vt;

	float s[3];

	// The SVD algorithm starts here
	float c, s_c, s_s;
	float c0, c1, c2;
	float s0, s1, s2;

	c0 = Ft[0][0]; c1 = Ft[0][1]; c2 = Ft[0][2];
	s0 = Ft[1][0]; s1 = Ft[1][1]; s2 = Ft[1][2];

	c = c0; s_c = s0;
	if (abs(s_c) < 1.0e-9f) { c0 = 1.f; s0 = 0.f; }
	else { float t = sqrt(c * c + s_c * s_c); c0 = c / t; s0 = s_c / t; }
	c = c1; s_c = s1;
	c1 = c0 * c + s0 * s_c;
	s1 = -s0 * c + c0 * s_c;
	c = c2; s_c = s2;
	c2 = c0 * c + s0 * s_c;
	s2 = -s0 * c + c0 * s_c;

	c = s1; s_c = Ft[2][1];
	if (abs(s_c) < 1.0e-9f) { s1 = 1.f; Ft[2][1] = 0.f; }
	else { float t = sqrt(c * c + s_c * s_c); s1 = c / t; Ft[2][1] = s_c / t; }
	c = s2; s_c = Ft[2][2];
	s2 = s1 * c + Ft[2][1] * s_c;
	Ft[2][2] = -Ft[2][1] * c + s1 * s_c;
	c = c2; s_c = Ft[2][0];
	c2 = s1 * c + Ft[2][1] * s_c;
	Ft[2][0] = -Ft[2][1] * c + s1 * s_c;

	Vt[0][0] = c0; Vt[0][1] = s0; Vt[0][2] = 0.f;
	Vt[1][0] = -s0 * s1; Vt[1][1] = c0 * s1; Vt[1][2] = -Ft[2][1];
	Vt[2][0] = s0 * Ft[2][1]; Vt[2][1] = -c0 * Ft[2][1]; Vt[2][2] = s1;

	s[0] = c1; s[1] = s2; s[2] = Ft[2][0];

	// Initialize U as identity matrix before accumulation
	U = glm::mat3(1.0f);

	for (int i = 0; i < 9; i++)
	{
		c = s[0]; s_c = s[1];
		if (abs(s_c) < 1.0e-9f) { c0 = 1.f; s0 = 0.f; }
		else { float t = sqrt(c * c + s_c * s_c); c0 = c / t; s0 = s_c / t; }
		s[0] = c0 * c + s0 * s_c;
		s[1] = -s0 * c + c0 * s_c;
		c = c2; s_c = s[2];
		c2 = c0 * c + s0 * s_c;
		s[2] = -s0 * c + c0 * s_c;

		glm::mat3 R_mat(glm::vec3(c0, -s0, 0), glm::vec3(s0, c0, 0), glm::vec3(0, 0, 1));
		U = U * R_mat;

		c = s[0]; s_c = c2;
		if (abs(s_c) < 1.0e-9f) { c0 = 1.f; s0 = 0.f; }
		else { float t = sqrt(c * c + s_c * s_c); c0 = c / t; s0 = s_c / t; }
		s[0] = c0 * c + s0 * s_c;
		c2 = -s0 * c + c0 * s_c;
		c = s[1]; s_c = s[2];
		s[1] = c0 * c + s0 * s_c;
		s[2] = -s0 * c + c0 * s_c;

		glm::mat3 R_mat2(glm::vec3(c0, 0, -s0), glm::vec3(0, 1, 0), glm::vec3(s0, 0, c0));
		U = U * R_mat2;
	}

	Sigma = glm::vec3(s[0], s[1], c2);

	V = glm::transpose(Vt);

	// --------------------------------------------------------------------------
	// --- START OF CORRECTION: Ensure proper rotation matrix ---
	// --------------------------------------------------------------------------

	// Make sure Sigma values are positive
	if (Sigma.x < 0) { Sigma.x = -Sigma.x; V[0][0] = -V[0][0]; V[1][0] = -V[1][0]; V[2][0] = -V[2][0]; }
	if (Sigma.y < 0) { Sigma.y = -Sigma.y; V[0][1] = -V[0][1]; V[1][1] = -V[1][1]; V[2][1] = -V[2][1]; }
	if (Sigma.z < 0) { Sigma.z = -Sigma.z; V[0][2] = -V[0][2]; V[1][2] = -V[1][2]; V[2][2] = -V[2][2]; }

	// Check for reflection and correct it
	// R = U * V^T, so we check det(U) and det(V)
	if (glm::determinant(U) * glm::determinant(V) < 0.0f)
	{
		// Invert the sign of the column of U corresponding to the smallest singular value
		// This flips the sign of det(U) while minimally affecting the matrix
		U[0][2] *= -1.0f;
		U[1][2] *= -1.0f;
		U[2][2] *= -1.0f;
	}
	// --------------------------------------------------------------------------
	// --- END OF CORRECTION ---
	// --------------------------------------------------------------------------
}
__device__ void computeCov3D_withMatrix(const glm::mat3& S_final, const glm::vec4& rot, float* cov3D) {
	glm::mat3 R = quatToMat3(rot);
	glm::mat3 M = S_final * R;
	glm::mat3 Sigma = glm::transpose(M) * M;

	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[1][1];
	cov3D[2] = Sigma[2][2];
	cov3D[3] = Sigma[0][1];
	cov3D[4] = Sigma[0][2];
	cov3D[5] = Sigma[1][2];
}
void eigenDecomposition(const Eigen::Matrix3f& A,
	Eigen::Vector3f& eigenValues,
	Eigen::Matrix3f& eigenVectors)
{
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(A);

	if (solver.info() != Eigen::Success) {
		throw std::runtime_error("Eigen decomposition failed!");
	}

	eigenValues = solver.eigenvalues();   // 고유값
	eigenVectors = solver.eigenvectors();  // 고유벡터 (정규직교화된 기저)
}
// device-side Jacobi for symmetric 3x3
// A_in: row-major symmetric matrix (A_in[i][j])
// outputs: eigenValues (ascending), eigenVectors (columns are eigenvectors)

// glm 매트릭스가 column-major이고 접근은 M[col][row] 이므로 변환에 주의.
// 여기서는 A_in_rowmajor[i][j] = glmM[j][i] 로 복사 (행-열 맞춤)

#define DEBUG_TARGET_IDX 44986 
#define COMPILE_TIME_MAX_K 10 
// device 함수: 3x3 역행렬 (adjoint 방식) + 성공 여부 반환
__device__ bool inverse3x3_safe(const glm::mat3& m, glm::mat3& inv, float eps_det = 1e-12f)
{
	float a00 = m[0][0], a01 = m[0][1], a02 = m[0][2];
	float a10 = m[1][0], a11 = m[1][1], a12 = m[1][2];
	float a20 = m[2][0], a21 = m[2][1], a22 = m[2][2];

	float det = a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20);
	
	if (fabs(det) < eps_det) return false;
	float invdet = 1.0f / det;

	inv[0][0] = (a11 * a22 - a12 * a21) * invdet;
	inv[0][1] = -(a01 * a22 - a02 * a21) * invdet;
	inv[0][2] = (a01 * a12 - a02 * a11) * invdet;

	inv[1][0] = -(a10 * a22 - a12 * a20) * invdet;
	inv[1][1] = (a00 * a22 - a02 * a20) * invdet;
	inv[1][2] = -(a00 * a12 - a02 * a10) * invdet;

	inv[2][0] = (a10 * a21 - a11 * a20) * invdet;
	inv[2][1] = -(a00 * a21 - a01 * a20) * invdet;
	inv[2][2] = (a00 * a11 - a01 * a10) * invdet;

	return true;
}
__device__ Eigen::Matrix3f invert3x3(const Eigen::Matrix3f& m) {
	float det =
		m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1)) -
		m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0)) +
		m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));

	float invdet = 1.0f / (abs(det) > 1e-8f ? det : 1e-8f);

	Eigen::Matrix3f inv;
	inv(0, 0) = (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1)) * invdet;
	inv(0, 1) = -(m(0, 1) * m(2, 2) - m(0, 2) * m(2, 1)) * invdet;
	inv(0, 2) = (m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1)) * invdet;
	inv(1, 0) = -(m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0)) * invdet;
	inv(1, 1) = (m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0)) * invdet;
	inv(1, 2) = -(m(0, 0) * m(1, 2) - m(0, 2) * m(1, 0)) * invdet;
	inv(2, 0) = (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0)) * invdet;
	inv(2, 1) = -(m(0, 0) * m(2, 1) - m(0, 1) * m(2, 0)) * invdet;
	inv(2, 2) = (m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0)) * invdet;

	return inv;
}
__device__ glm::mat3 polarDecomposition(const glm::mat3& A)
{
	// 1. A로부터 대칭 행렬 AtA (= S*S)를 계산합니다.
	glm::mat3 AtA = glm::transpose(A) * A;

	// 2. AtA를 고유값 분해합니다. (AtA = V * D * V_transpose)
	glm::vec3 eigenvalues;      // D (대각성분)
	glm::mat3 eigenvectors_V; // V
	eigenDecomposition_glm(AtA, eigenvalues, eigenvectors_V);

	// 3. 스트레칭 행렬 S를 계산합니다. S = V * sqrt(D) * V_transpose
	glm::mat3 D_sqrt = glm::mat3(1.0f);
	// 0보다 작은 eigenvalue에 sqrt를 적용하는 것을 방지
	D_sqrt[0][0] = sqrtf(fmaxf(0.0f, eigenvalues.x));
	D_sqrt[1][1] = sqrtf(fmaxf(0.0f, eigenvalues.y));
	D_sqrt[2][2] = sqrtf(fmaxf(0.0f, eigenvalues.z));

	glm::mat3 S = eigenvectors_V * D_sqrt * glm::transpose(eigenvectors_V);

	// 4. 회전 행렬 R을 계산합니다. R = A * S의 역행렬
	glm::mat3 S_inv = glm::inverse(S);
	glm::mat3 R = A * S_inv;

	// (안정성을 위해) 계산된 R의 determinant가 음수이면 반전된 것이므로 부호를 바꿔줍니다.
	if (glm::determinant(R) < 0.0f) {
		R *= -1.0f;
	}

	// --- 수정된 부분 ---
	// 순수 회전 R과 순수 스트레칭 S를 곱하여
	// 불안정한 요소가 제거된 '깨끗한 A'를 반환합니다.
	return R * S;
}

// ── [SH] 순수 회전만 뽑아내기 (Higham 반복) ───────────────────────────
//  A = R·S 의 R 만 필요하다. 위 polarDecomposition 은 glm::inverse(S) 를 쓰는데
//  납작한 변형에서 S 가 특이해져 터진다. 반복법은 역행렬을 A 자체에 대해서만
//  쓰므로 그런 문제가 없고, 특이값 클램프를 거치지 않아 결과가 '진짜 회전'이다.
//      R ← ½ ( R + R⁻ᵀ )   를 반복하면 극분해의 회전 인자로 2차 수렴한다.
//  실패(축퇴·반전) 시 false 를 돌려주고 호출부는 회전을 적용하지 않는다.
__device__ __forceinline__ bool polarRotationGLM(const glm::mat3& A, glm::mat3& R)
{
	R = A;
	for (int it = 0; it < 5; ++it) {
		const float d = glm::determinant(R);
		if (!isfinite(d) || fabsf(d) < 1e-20f) return false;
		R = 0.5f * (R + glm::transpose(glm::inverse(R)));
	}
	// 반전(det<0)은 회전이 아니다. 억지로 부호를 뒤집으면 하이라이트가 튀므로 포기한다.
	const float dR = glm::determinant(R);
	return isfinite(dR) && dR > 0.0f;
}
// Eigen을 활용한 아주 깔끔한 반복 극분해 함수
__device__ void extractRotationIterativeEigen(const Eigen::Matrix3f& A, Eigen::Matrix3f& R) {


	// 1. 행렬 A의 크기(Frobenius Norm) 구하기
	float sq_norm = 0.0f;
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			sq_norm += A(i, j) * A(i, j);
		}
	}
	float norm = sqrtf(sq_norm);

	// 2. 납작한 노이즈이거나 크기가 너무 작으면 "회전 포기! (단위 행렬)"
	if (norm < 1e-10f) {
		R = Eigen::Matrix3f::Identity();
		return;
	}

	// 3. 핵심 비법: 스케일을 1.0으로 뻥튀기해서 반복문의 수렴 속도와 안정성을 극대화!
	R = A / norm;

	for (int iter = 0; iter < 10; ++iter) {
		float det =
			R(0, 0) * (R(1, 1) * R(2, 2) - R(1, 2) * R(2, 1)) -
			R(0, 1) * (R(1, 0) * R(2, 2) - R(1, 2) * R(2, 0)) +
			R(0, 2) * (R(1, 0) * R(2, 1) - R(1, 1) * R(2, 0));

		// 중간에 행렬이 찌그러지면 회전을 멈추고 단위 행렬 반환
		if (abs(det) < 1e-8f) {
			R = Eigen::Matrix3f::Identity();
			return;
		}

		Eigen::Matrix3f invR = invert3x3(R);
		R = 0.5f * (R + invR.transpose());
	}


	//R = A; // 초기 R을 A로 설정
	//
	//for (int iter = 0; iter < 10; ++iter) {
	//	// 1. 역행렬 구하기 (상민님의 invert3x3 재활용!)
	//	Eigen::Matrix3f invR = invert3x3(R);
	//
	//	// 2. R_{next} = 0.5 * (R + (invR)^T) 
	//	// Eigen의 transpose() 덕분에 코드가 예술적으로 짧아집니다.
	//	R = 0.5f * (R + invR.transpose());
	//}
}
//// 16개의 이웃(Neighbor) 데이터를 한 번에 처리하여 PPt, QPt를 계산하는 커널
//__global__ void compute_deformation_gradient_tensorcore(
//	// ... 인자들 ...
//	int idx // 현재 처리 중인 가우시안 인덱스
//) {
//	// 워프 설정
//	int warpId = threadIdx.x / 32;
//	int laneId = threadIdx.x % 32;
//
//	// 1. Shared Memory에 이웃 데이터 로딩
//	// Matrix P (16x3): 16개 이웃의 원래 위치 상대좌표
//	// Matrix Q (16x3): 16개 이웃의 변형된 위치 상대좌표
//	__shared__ half smem_P[16 * 16];
//	__shared__ half smem_Q[16 * 16];
//	__shared__ float smem_PPt[16 * 16]; // 결과 1
//	__shared__ float smem_QPt[16 * 16]; // 결과 2
//
//	// laneId 0~15가 각각 이웃 0~15번 데이터를 로딩
//	if (laneId < 16) {
//		// nbr_index에서 이웃 가져오기
//		int nbr_idx = nbr_index[idx * MAX_K + laneId]; // MAX_K는 16 이상이어야 함
//
//		glm::vec3 p_vec = original_pos[nbr_idx] - center_original;
//		glm::vec3 q_vec = deformed_pos[nbr_idx] - center_deformed;
//
//		// P 행렬 채우기 (Row Major)
//		smem_P[laneId * 16 + 0] = __float2half(p_vec.x);
//		smem_P[laneId * 16 + 1] = __float2half(p_vec.y);
//		smem_P[laneId * 16 + 2] = __float2half(p_vec.z);
//		// 나머지 0 패딩...
//
//		// Q 행렬 채우기
//		smem_Q[laneId * 16 + 0] = __float2half(q_vec.x);
//		smem_Q[laneId * 16 + 1] = __float2half(q_vec.y);
//		smem_Q[laneId * 16 + 2] = __float2half(q_vec.z);
//	}
//
//	// Matrix P Transpose (PPt 계산을 위해 필요)
//	// 텐서 코어는 A * B^T 지원 (col_major 로딩 시)
//
//	// 2. 텐서 코어 연산 수행
//	wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> p_frag; // P Transpose 처럼 동작
//	wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> p_frag_b;
//	wmma::fragment<wmma::accumulator, 16, 16, 16, float> ppt_frag;
//
//	wmma::fill_fragment(ppt_frag, 0.0f);
//	wmma::load_matrix_sync(p_frag, smem_P, 16);   // P^T
//	wmma::load_matrix_sync(p_frag_b, smem_P, 16); // P
//
//	// PPt = P^T * P (3x3 결과가 나옴)
//	// 원래는 Outer Product 합인데, 행렬 곱셈 형태로 변환
//	wmma::mma_sync(ppt_frag, p_frag, p_frag_b, ppt_frag);
//
//	// QPt 계산 (Q * P^T)
//	wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> q_frag;
//	wmma::fragment<wmma::accumulator, 16, 16, 16, float> qpt_frag;
//
//	wmma::fill_fragment(qpt_frag, 0.0f);
//	wmma::load_matrix_sync(q_frag, smem_Q, 16);
//
//	wmma::mma_sync(qpt_frag, q_frag, p_frag_b, qpt_frag);
//
//	// 3. 결과 저장 및 A 행렬 계산
//	wmma::store_matrix_sync(smem_PPt, ppt_frag, 16, wmma::mem_row_major);
//	wmma::store_matrix_sync(smem_QPt, qpt_frag, 16, wmma::mem_row_major);
//
//	// 이후 스레드 0번이 3x3 역행렬 계산 및 A = QPt * inv(PPt) 수행
//}






template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const float* realOrigin_points,       
	// chainmail 후 (매 프레임 갱신)
	// --- neighborhood for 3D F ---
	// 이웃 그래프는 물리 커널이 쓰는 것과 동일한 CSR 버퍼를 그대로 읽는다.
	// 가우시안마다 차수가 다르므로 고정 stride가 존재하지 않는다.
	const int* nbr_offset,   // size: P
	const int* nbr_count,    // size: P
	const int* nbr_idx,      // size: 방향성 이웃 총 개수
	const float* nbr_time,

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
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
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
	bool _bubble,
	// Step 2 — J 시각화: nullptr이 아니면 SH 색 대신 J 컬러맵으로 렌더한다.
	// 파랑 J<1(압축) ← 흰색 J=1 → 빨강 J>1(팽창). visualization convention.
	const float* volJ,
	float jVizGain,
	// matType 시각화: nullptr이 아니면 물질 분류를 색으로 렌더한다.
	// volume=회색, surface=파랑, fiber=빨강, 이웃부족=어두운 회색, 수치실패=자홍
	const int* volMatType
	)
{
	const float* cov3D;

	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;
	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	
	// 1. 좌표 정보 가져오기 (올바른 매핑)
	float3 p_original = { realOrigin_points[3 * idx], realOrigin_points[3 * idx + 1], realOrigin_points[3 * idx + 2] };  // 변형 전
	float diff = glm::length(glm::vec3(p_orig.x, p_orig.y, p_orig.z) -
		glm::vec3(p_original.x, p_original.y, p_original.z));

	

	if (p_orig.x < boxmin.x || p_orig.y < boxmin.y || p_orig.z < boxmin.z ||
		p_orig.x > boxmax.x || p_orig.y > boxmax.y || p_orig.z > boxmax.z)
		return;



	float rx = glm::radians(_pivotRotX);
	float ry = glm::radians(_pivotRotY);
	float rz = glm::radians(_pivotRotZ);


	//회전행렬
	glm::mat4 RX = glm::mat4(
		1, 0, 0, 0,
		0, cos(rx), -sin(rx), 0,
		0, sin(rx), cos(rx), 0,
		0, 0, 0, 1);
	glm::mat4 RY = glm::mat4(
		cos(ry), 0, sin(ry), 0,
		0, 1, 0, 0,
		-sin(ry), 0, cos(ry), 0,
		0, 0, 0, 1);
	glm::mat4 RZ = glm::mat4(
		cos(rz), -sin(rz), 0, 0,
		sin(rz), cos(rz), 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1);
	glm::mat4 QR = RZ * RY * RX;
	
	glm::vec3 pivot = glm::vec3(0.5f,-0.001f,0.0f);
	float dx = p_orig.x - pivot.x; 
	float dy = p_orig.y - pivot.y; 
	float dz = p_orig.z - pivot.z; 
	float dist2 = dx * dx + dy * dy + dz * dz; 
	float radius = 3.0f;
	glm::vec4 Rot; 
	bool applyRotation = dist2 < radius* radius;
	// 중심 (0,0,0) 기준 ROI 범위
	// 중심
	// ===============================
// (0,0,0) 기준 국소 ROI 바운딩 박스
// ===============================
	if (/*applyRotation*/true) { 
		// 좌표 변환 
		glm::vec4 p4 = glm::vec4(p_orig.x, p_orig.y, p_orig.z, 1.0f); 
		p4 = QR * p4;
		p_orig = make_float3(p4.x, p4.y, p4.z);
	}
	

	
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	// 0) 데이터 준비
	glm::vec3 xi = glm::vec3(orig_points[3 * idx + 0], orig_points[3 * idx + 1], orig_points[3 * idx + 2]);   // 현재(체인메일 후)
	glm::vec3 xi0 = glm::vec3(realOrigin_points[3 * idx + 0], realOrigin_points[3 * idx + 1], realOrigin_points[3 * idx + 2]); // 초기

	const bool has_graph = (nbr_offset != nullptr && nbr_count != nullptr && nbr_idx != nullptr);
	const int nbr_off = has_graph ? nbr_offset[idx] : 0;
	const int kn = has_graph ? nbr_count[idx] : 0;
	const float deform_eps = 1e-2f;


	


	/////////////////////////////////////////////////////////////////////////////////
	glm::vec3 final_new_scales = scales[idx]; // 기본값은 현재 스케일
	glm::quat final_new_rotation(rotations[idx].x, rotations[idx].y, rotations[idx].z, rotations[idx].w); // 기본값은 원래 회전값으로 설정
	// << 1. 변환 행렬 A를 계산하는 로직 >>
	glm::mat3 A_final;
	glm::mat3 R;
	glm::mat3 S_final;
	glm::vec3 S_vec;
	bool ok = false;
	// [SH] 조명 회전 전용 '순수 회전'. 아래 SVD/클램프 경로와 완전히 독립이다.
	glm::mat3 R_sh(1.0f);
	bool shRotOK = false;
	// ── [TN] 접선-법선 분해 경로 (g_tnMode=1). 성공하면 cov3D 를 여기서 바로 쓰고
	//    아래 기존 3D LS 블록 전체를 건너뛴다. 실패하면 tnWrote=false 로 기존 경로 그대로. ──
	bool tnWrote = false;
	if (g_tnMode != 0 && _twist && diff > deform_eps && kn >= 3 && has_graph) {
		tnWrote = computeTangentNormalCov(idx, P, realOrigin_points, orig_points,
			nbr_idx, nbr_off, kn, scales[idx], rotations[idx], scale_modifier,
			cov3Ds + idx * 6);
		atomicAdd(&g_tnUseCtr[tnWrote ? 0 : 1], 1u);
	}
	if (tnWrote)
	{
		// 이미 계산됨 — 기존 경로 스킵 (속도 비교가 공정하도록 중복 계산하지 않는다)
	}
	else if (kn < 3)
	{
		// A 계산을 시도하지 않고, 변형이 없는 항등 행렬을 반환
		A_final = glm::mat3(1.0f);
		ok = false;
		atomicAdd(&g_renderFCtr[2], 1u);   // 실패 원인 구분: det 실패와 이웃부족
	} 
	else
	{
		glm::mat3 PPt(0.0f);
		glm::mat3 QPt(0.0f);
		//glm::vec3 delta_vectors[COMPILE_TIME_MAX_K];
		//int valid_delta_count = 0;
		glm::vec3 original_pos = glm::vec3(
			realOrigin_points[3 * idx], 
			realOrigin_points[3 * idx + 1], 
			realOrigin_points[3 * idx + 2]);
		glm::vec3 deformed_pos = glm::vec3(
			orig_points[3 * idx], 
			orig_points[3 * idx + 1], 
			orig_points[3 * idx + 2]);
		const float sigma_sq = 0.01f;
		// 토글 값은 루프 불변 — 이웃마다 전역을 다시 읽지 않도록 밖에서 1회만 읽는다.
		// (OFF일 때 기존 경로의 내부 루프가 완전히 예전 형태로 돌아간다)
		const bool useCushion = (g_volCovReg != 0);
		const float cushionLambda = g_volCovLambda;
		for (int i = 0; i < kn; ++i) {
			int neighbor_idx = nbr_idx[nbr_off + i];
			if (neighbor_idx < 0 || neighbor_idx >= P)
				continue;
			glm::vec3 nbr_original_pos = glm::vec3(
				realOrigin_points[3 * neighbor_idx],
				realOrigin_points[3 * neighbor_idx + 1],
				realOrigin_points[3 * neighbor_idx + 2]);
			glm::vec3 nbr_deformed_pos = glm::vec3(
				orig_points[3 * neighbor_idx],
				orig_points[3 * neighbor_idx + 1],
				orig_points[3 * neighbor_idx + 2]);
			// 그대로 행렬에 삽입
			
			  // 중심 기준 상대 좌표 (delta 벡터)
			glm::vec3 p = nbr_original_pos - original_pos;
			glm::vec3 q = nbr_deformed_pos - deformed_pos;

			//glm::vec3 p = nbr_original_pos;
			//glm::vec3 q = nbr_deformed_pos;

			// 2. 벡터 변화 확인 (늘어났는지 줄어들었는지)
		
			PPt += glm::outerProduct(p, p); // P * P^T
			QPt += glm::outerProduct(q, p); // Q * P^T
			// ── 젤리 쿠션(공분산 인지 정규화, 실시간 토글 g_volCovReg) ──
			//  이웃을 '점'이 아니라 '두께 있는 덩어리'로: 이웃의 rest 공분산을 PPt에 더해
			//  중심점만으론 비어 있던 방향(평면 법선)을 이웃 자신의 두께로 채운다.
			//  OFF(=0)면 이 블록을 통째로 건너뛰어 기존 경로와 완전히 동일.
			if (useCushion) {
				float cAA[6];
				computeCov3D(scales[neighbor_idx], 1.0f, rotations[neighbor_idx], cAA);
				glm::mat3 Sj(cAA[0], cAA[1], cAA[2],
					cAA[1], cAA[3], cAA[4],
					cAA[2], cAA[4], cAA[5]);
				const glm::mat3 cush = cushionLambda * Sj;
				PPt += cush; QPt += cush;   // ★ 양쪽에 더해야 rest 에서 A=I (축소 버그 수정)
			}
		}
		
		//computePPtQPt_ptx(p_list, q_list, kn, PPt, QPt);
		// 안정성을 위해 PPt에 작은 값을 더해줌 (레귤러라이제이션)
		float tracePPt = PPt[0][0] + PPt[1][1] + PPt[2][2];
		// 시스템 에너지의 0.01%~0.1% 정도만 보정값으로 사용합니다.
		// 이렇게 하면 가우시안이 아주 작아도 그에 맞춰 보정값이 작아집니다.
		float adaptive_alpha = (tracePPt > 0.0f) ? (tracePPt * 1e-4f) : 1e-6f;
		PPt += glm::mat3(adaptive_alpha);
		// ★ 분모에만 더하면 rest 에서 A≠I 가 되어 얇은 방향이 λ/(λ+ε) 로 눌린다.
		//   양쪽에 더해야 rest 에서 A=I 이고, 관측 불가 방향이 '변형 없음'으로 남는다.
		if (g_renderFRobust) QPt += glm::mat3(adaptive_alpha);
		//const float regularization_alpha = 1e-5f;
		//PPt += glm::mat3(regularization_alpha);
		// === [핵심 수정 2] 레귤러라이제이션 값 조정 ===
		//const float active_regularization = (kn < 6) ? 1e-3f : 1e-5f; // 이웃 개수가 적을 때 더 강하게
		//PPt += glm::mat3(active_regularization);

		glm::mat3 invPPt;
		if (g_renderFRobust) {
			// det(PPt) 는 [길이⁶] 이라 절대 임계값은 씬 스케일과 함께 움직인다.
			// trace 로 스케일을 분리하면 남는 det 는 '모양'만 나타내는 O(1) 값이 된다.
			//   PPt = sP·P̂,  trace(P̂) = 3   ⟹   P̂ 는 등방이면 det=1, 납작할수록 0
			//   (sP·P̂)⁻¹ = P̂⁻¹ / sP
			// 임계값 1e-9: ε=1e-4·trace 정규화 후 최악(직선형) det̂ ≈ 2.7e-7 이라
			// 270배 여유. 진짜 축퇴(전부 한 점)만 걸러낸다.
			const float sP = fmaxf((PPt[0][0] + PPt[1][1] + PPt[2][2]) * (1.0f / 3.0f), 1e-30f);
			const float invSP = 1.0f / sP;
			glm::mat3 invHat;
			ok = inverse3x3_safe(PPt * invSP, invHat, 1e-9f);
			if (ok) invPPt = invHat * invSP;
		}
		else {
			ok = inverse3x3_safe(PPt, invPPt, 1e-8f);
		}
		atomicAdd(&g_renderFCtr[ok ? 0 : 1], 1u);
		glm::mat3 A;
		
		if (ok) {
			
			A = QPt * invPPt; //glm::inverse(PPt);
			// ★ [SH] 조명용 회전은 여기서 A 로부터 곧바로 뽑는다.
			//   아래 SVD → 특이값 클램프 → S_inv 경로를 타지 않으므로 클램프 오염이 없다.
			// ⚠️ 모양 경로와 '완전히 같은 조건'에서만 적용해야 한다.
			//    아래 공분산 적용부는 (_twist && diff > deform_eps && ok) 일 때만
			//    변형을 반영하고, 아니면 원래 회전을 유지한다. 여기에 같은 게이트를
			//    걸지 않으면 rest 나 _wave 경로에서 '모양은 그대로인데 색만 도는'
			//    불일치가 생긴다(실제로 rest 에서 미세한 색 변화로 관측됨).
			if (g_shRotate != 0 && _twist && diff > deform_eps) {
				glm::mat3 Rp;
				if (polarRotationGLM(A, Rp)) { R_sh = Rp; shRotOK = true; }
			}
			 // 2. A를 SVD를 통해 U, S, V로 분해
			glm::mat3 AtA = glm::transpose(A) * A;
			glm::mat3 V;
			glm::vec3 S_squared;
			eigenDecomposition_glm(AtA, S_squared, V);
			S_vec = glm::sqrt(glm::max(glm::vec3(0.0f), S_squared));


			const float epsilon = 1.0e-7f;
			const float min_s = 1e-3f;
			S_vec = glm::max(S_vec, glm::vec3(min_s));
			const float max_s = 1.5f;
			S_vec = glm::min(S_vec, glm::vec3(max_s));
			glm::vec3 S_inv_vec = glm::vec3(
				(S_vec.x > epsilon) ? 1.0f / S_vec.x : 0.0f,
				(S_vec.y > epsilon) ? 1.0f / S_vec.y : 0.0f,
				(S_vec.z > epsilon) ? 1.0f / S_vec.z : 0.0f);
			//const float min_singular_value = 1e-4f;
			//if (S_vec.x < min_singular_value) S_vec.x = 1.0f;
			//if (S_vec.y < min_singular_value) S_vec.y = 1.0f;
			//if (S_vec.z < min_singular_value) S_vec.z = 1.0f;
			//// --- 여기까지 수정 ---
			//
			//// 이제 안정화된 S_vec으로 S_inv_vec를 계산
			//glm::vec3 S_inv_vec = glm::vec3(1.0f / S_vec.x, 1.0f / S_vec.y, 1.0f / S_vec.z);
			glm::mat3 U =
				A * V *
				glm::mat3(
					glm::vec3(S_inv_vec.x, 0, 0),
					glm::vec3(0, S_inv_vec.y, 0),
					glm::vec3(0, 0, S_inv_vec.z));

			// 3. 순수 회전(R)과 순수 스케일(S_vec) 분리
			R = U * glm::transpose(V);
			if (glm::determinant(R) < 0.0f) {
				U[0] *= -1.0f;
				R = U * glm::transpose(V);
			}
			// R의 행렬식이 음수일 때, 가장 수치적 영향이 적은(가장 작은 특이값) 축을 보정
			//if (glm::determinant(R) < 0.0f) {
			//	int smallest_idx = 0;
			//	if (S_vec[1] < S_vec[smallest_idx]) smallest_idx = 1;
			//	if (S_vec[2] < S_vec[smallest_idx]) smallest_idx = 2;
			//
			//	U[smallest_idx] *= -1.0f; // 가장 작은 축의 부호를 반전
			//	R = U * glm::transpose(V);
			//}

			// 5. '순수 회전'과 '확대만 남은 스케일'로 새로운 변환 행렬 A_final을 재조립
			// A_final = R * S (Polar Decomposition)
			glm::mat3 S_final_diag = glm::mat3(
				glm::vec3(S_vec.x, 0, 0),
				glm::vec3(0, S_vec.y, 0),
				glm::vec3(0, 0, S_vec.z));
			S_final = V * S_final_diag * glm::transpose(V);

			A_final = S_final * R;
		}
		else {
			// fallback: identity 또는 작게 블렌딩
			A = glm::mat3(1.0f);
		}
	}

	glm::mat3 s_base(1.0f);
	float final_opacity = opacities[idx];


	if (tnWrote) {
		// [TN] cov3D 이미 기록됨 — 아래 기존 공분산 경로들을 모두 건너뛴다.
	}
	else if (_twist && diff > deform_eps&& ok) {
		if (_bubble) {
			S_final = glm::mat3(1.0f);
		}
		computeCov3D2(
			_bubble,
			scales[idx],
			R,                  // 변형에서 얻은 회전
			S_final,              // 변형에서 얻은 스케일
			A_final,
			scale_modifier,
			rotations[idx],
			cov3Ds + idx * 6);
	}else if (_wave) {
		
		// 원래 회전 쿼터니언 
		glm::quat originalRot(rotations[idx].x, rotations[idx].y, rotations[idx].z, rotations[idx].w); 
		// 새 회전 쿼터니언 생성 
		glm::quat pivotRot = glm::quat(glm::mat3(QR));
		// 두 회전 결합 (왼쪽에서 오른쪽으로 적용) 
		glm::quat newRot = pivotRot * originalRot; 
		// 정규화
		newRot = glm::normalize(newRot); 
		// 수정된 회전으로 공분산 행렬 계산 
		//glm::quat ->> w x y z 순서 
		glm::vec4 modifiedRot(newRot.w, newRot.x, newRot.y, newRot.z);
		computeCov3D(
			scales[idx], 
			scale_modifier,
			modifiedRot,
			cov3Ds + idx * 6);
	}
	else {
		// 변형이 거의 없으면 기존 값 유지
		computeCov3D(
			scales[idx],
			scale_modifier, 
			rotations[idx],
			cov3Ds + idx * 6);
	}

	cov3D = cov3Ds + idx * 6;
		
		// If 3D covariance matrix is precomputed, use it, otherwise compute
		// from scaling and rotation parameters. 
	
	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix, 
		
		_rotatingModifier_COV2D_Matrix_x,
		_rotatingModifier_COV2D_Matrix_y,
		_rotatingModifier_COV2D_Matrix_z


		);

	constexpr float h_var = 0.3f;
	const float det_cov = cov.x * cov.z - cov.y * cov.y;
	//가우시안이 카메라에서 너무 멀어지면 화면상에서 1픽셀보다 작아져 반짝거리는 현상(Aliasing)이 생김. 
	//따라서 행렬대각 성분(x, z)에 강제로 0.3 분산더해 최소한의 크기보장(고전적인 EWA Splatting 기법)
	cov.x += h_var;
	cov.z += h_var;
	const float det_cov_plus_h_cov = cov.x * cov.z - cov.y * cov.y;
	float h_convolution_scaling = 1.0f;

	if(antialiasing)
		h_convolution_scaling = sqrt(max(0.000025f, det_cov / det_cov_plus_h_cov)); // max for numerical stability

	// Invert covariance (EWA algorithm)
	const float det = det_cov_plus_h_cov;

	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	//투영된 가우시안이 화면에서 얼마나 크게 퍼지는지 최대 반지름my_radius
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	//p_proj = -1.0 ~ 1.0 사이의 정규화된 가우시안 중심 좌표(NDC)
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };//픽셀 해상도(예: 1920x1080)에 맞게 곱해서, 화면 상의 정확한 픽셀 가우시안 좌표point_image로 변환
	uint2 rect_min, rect_max;


	//반지름(my_radius)을 이용해, 화면을 나눈 16X16 픽셀 타일 상 이 가우시안이 최소 몇 번 타일부터 최대 몇 번 타일 덮는지(Bounding Box)를 계산.
	//가우시안중심점(point_image)에서 상하좌우로 반지름(my_radius)만큼 사각형(Bounding Box).
	//그리고 이 사각형이 타일 그리드(Tile Grid) 상에서 어디에 걸치는지 계산.
    //rect_min: 사각형이 걸친 가장 왼쪽 위 타일의 인덱스(ex x방향 10번째, y방향 5번째 타일  x = 10, y = 5)
	//rect_max : 사각형이 걸친 가장 오른쪽 아래 타일의 인덱스(ex x방향 13번째, y방향 8번째 타일  x = 13, y = 8)
	if (rects == nullptr) 	// More conservative
	{
		getRect(point_image, my_radius, rect_min, rect_max, grid);
	}
	else // Slightly more aggressive, might need a math cleanup
	{
		const int2 my_rect = { (int)ceil(3.f * sqrt(cov.x)), (int)ceil(3.f * sqrt(cov.z)) };
		rects[idx] = my_rect;
		getRect(point_image, my_rect, rect_min, rect_max, grid);
	}
	//화면을 아예 벗어났거나 가리는 타일이 0개라면 더 이상 계산할 필요 없이 버림 return
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		// [SH] 회전이 유효할 때만 넘긴다. nullptr 이면 기존 경로와 비트 단위로 동일하다.
		const glm::mat3* shRotPtr = (g_shRotate != 0 && shRotOK) ? &R_sh : nullptr;
		if (g_shRotate != 0) atomicAdd(&g_shRotCtr[shRotOK ? 0 : 1], 1u);
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped, shRotPtr);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// ── Step 2: J 시각화 ──
	// 각 가우시안의 부피비 J를 색으로 덮어쓴다: 파랑 J<1(눌림), 흰색 J=1, 빨강 J>1(부풂).
	// t = (J-1)·gain 을 [-1,1]로 클램프 → gain=3이면 ±33% 변화에서 색이 포화된다.
	// (matType이 volume이 아닌 가우시안은 J 버퍼가 1.0이라 흰색으로 나온다)
	if (volJ != nullptr)
	{
		const float Jv = volJ[idx];
		float t = (isfinite(Jv) ? (Jv - 1.0f) : 0.0f) * jVizGain;
		t = fminf(1.0f, fmaxf(-1.0f, t));
		if (t < 0.0f) {
			// 흰색 → 파랑 (압축)
			const float k = -t;
			rgb[idx * C + 0] = 1.0f - 0.85f * k;
			rgb[idx * C + 1] = 1.0f - 0.60f * k;
			rgb[idx * C + 2] = 1.0f;
		}
		else {
			// 흰색 → 빨강 (팽창)
			rgb[idx * C + 0] = 1.0f;
			rgb[idx * C + 1] = 1.0f - 0.60f * t;
			rgb[idx * C + 2] = 1.0f - 0.85f * t;
		}
	}

	// ── matType 시각화 ──
	// 모양 기반 물질 분류가 실제 재질(머리카락/피부/안경)과 맞는지 눈으로 확인한다.
	//   volume=회색(부피 지킴) / surface=파랑 / fiber=빨강 / 이웃부족=진회색 / 수치실패=자홍
	// 머리카락이 파랑·빨강으로 칠해지고 피부가 회색이면 분류가 재질과 맞는 것.
	if (volMatType != nullptr)
	{
		const int mt = volMatType[idx];
		float r, g, b;
		switch (mt) {
		case 0: r = 0.55f; g = 0.55f; b = 0.55f; break; // volume  회색
		case 1: r = 0.10f; g = 0.35f; b = 1.00f; break; // surface 파랑
		case 2: r = 1.00f; g = 0.15f; b = 0.10f; break; // fiber   빨강
		case 3: r = 0.20f; g = 0.20f; b = 0.20f; break; // 이웃부족 진회색
		default: r = 1.00f; g = 0.00f; b = 1.00f; break; // 수치실패 자홍
		}
		rgb[idx * C + 0] = r;
		rgb[idx * C + 1] = g;
		rgb[idx * C + 2] = b;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	float opacity = final_opacity;// opacities[idx];


	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacity * h_convolution_scaling };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);//총 몇 개의 타일을 덮었는지, 가로로 덮은 타일 수: 13 - 10 = 3개 세로로 덮은 타일 수 : 8 - 5 = 3개 총3x3=9개
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list, //타일별로 깊이 정렬된 가우시안 인덱스 리스트
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	int* __restrict__ id_buffer,
	// 체커 바닥 (ground_on = 0 이면 기존 배경 합성과 동일). 전부 값 전달 — 픽셀마다 전역 메모리를 읽지 않는다.
	int ground_on,
	float3 gr_cam,      // 카메라 위치 (world)
	float3 gr_rx,       // 카메라 x축의 world 표현 × tan_fovx  (view R 의 0행)
	float3 gr_ry,       // 카메라 y축(화면 아래) × tan_fovy
	float3 gr_rz,       // 카메라 z축(전방)
	float3 gr_n,        // 바닥 법선 (up)
	float3 gr_t1,       // 바닥 접선 / 체커 크기
	float3 gr_t2,
	float3 gr_origin,   // 체커 원점 (물체 아래 평면 위의 점)
	float gr_camH,      // n·cam − h  (> 0 이면 카메라가 바닥 위)
	float gr_pixScale,  // 픽셀 1개의 각크기 / 체커 크기 (AA 판정용)
	float gr_invFade2)  // 1 / fadeRadius²
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();//현재 블록 가져옴
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;//가로로 타일 총 개수 ex1920 / 16 = 120개의 타일
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };//우리 팀이 맡은 타일의 왼쪽 위(시작) 픽셀 좌표.  (1, 1)번 타일이면, 시작 픽셀은 (16, 16).
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };//타일의 오른쪽 아래(끝) 픽셀 좌표
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };//스레드가 칠해야 할 정확한 픽셀 좌표(x, y)
	uint32_t pix_id = W * pix.y + pix.x;//모니터 화면 전체를 1차원 배열로 쭉 폈을 때, 내 픽셀이 몇 번째 칸에 있는지(1D 인덱스)
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];//이 타일에 가우시안이 총 몇 개나 겹쳐 있는지, 사전(ranges 배열)에서 찾아오는 과정
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);//공유메모리크기 256 이니 가우시안이 그 이상이면 몇번의 rounds를 해야하는지 256개면 1rounds
	int toDo = range.y - range.x;//총 처리해야하는 가우시안 개수

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];//그걸 배치 단위로 collected_id[BLOCK_SIZE]에 가져와서 쓰는 것
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };
	int picked_id = -1;//각 픽셀에서 -1로 시작

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)//총 rounds 만큼반복 한 rounds에 256 개 가우시안 처리
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();//공유메모리에 올림 256 개

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)//256개 or 그보다 적은 가우시안이라면 가우시안 개수만큼 반복
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = 
				-0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y)
				- con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			if (id_buffer != nullptr && picked_id == -1 && alpha > 0.1f)
			{
				//front to back 순서의 가우시안을 돌며 0.1 보다 alpha 가 큰 첫 가우시안 id 를 저장
				picked_id = collected_id[j];//그리고 내부 루프 j는 이번 배치의 j번째 가우시안을 의미함
			}
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;

		// 체커 바닥: 남은 투과율 T로 보이는 '배경' 자리에 광선-평면 교점의 체커를 합성한다.
		// 물체는 접촉 제약으로 항상 바닥 위에 있으므로 깊이 비교 없이 배경으로 둬도 가림 순서가 맞다.
		float floorW = 0.0f;      // 바닥 가중치 (0 = 원래 배경)
		float floorShade = 0.5f;  // 0 = 어두운 칸, 1 = 밝은 칸
		if (ground_on != 0 && T > (1.0f / 255.0f) && gr_camH > 0.0f)
		{
			const float ndcx = (2.0f * pixf.x + 1.0f) / (float)W - 1.0f;
			const float ndcy = (2.0f * pixf.y + 1.0f) / (float)H - 1.0f;
			const float dx = gr_rx.x * ndcx + gr_ry.x * ndcy + gr_rz.x;
			const float dy = gr_rx.y * ndcx + gr_ry.y * ndcy + gr_rz.y;
			const float dz = gr_rx.z * ndcx + gr_ry.z * ndcy + gr_rz.z;
			const float denom = gr_n.x * dx + gr_n.y * dy + gr_n.z * dz;
			if (denom < -1e-6f)
			{
				const float s = -gr_camH / denom;   // dir의 카메라 z 성분이 1이라 s가 곧 뷰 깊이
				const float ox = gr_cam.x + s * dx - gr_origin.x;
				const float oy = gr_cam.y + s * dy - gr_origin.y;
				const float oz = gr_cam.z + s * dz - gr_origin.z;
				const float fade = exp(-(ox * ox + oy * oy + oz * oz) * gr_invFade2);
				// 페이드가 먼저라 수평선 근처의 거대한 u,v는 int 변환 전에 걸러진다.
				if (fade > (1.0f / 255.0f))
				{
					const float u = ox * gr_t1.x + oy * gr_t1.y + oz * gr_t1.z;   // 체커 칸 단위
					const float v = ox * gr_t2.x + oy * gr_t2.y + oz * gr_t2.z;
					const int parity = ((int)floorf(u) + (int)floorf(v)) & 1;
					// 픽셀 발자국(칸 단위) = 광선 거리 × 픽셀 각크기 / 입사각 cos.
					// 칸보다 커지면 두 색의 평균으로 수렴시켜 먼 바닥의 모아레를 없앤다.
					const float dl2 = dx * dx + dy * dy + dz * dz;
					const float aa = min(1.0f, s * dl2 * gr_pixScale / (-denom));
					floorShade = (parity ? 1.0f : 0.0f) * (1.0f - aa) + 0.5f * aa;
					floorW = fade;
				}
			}
		}
		for (int ch = 0; ch < CHANNELS; ch++)
		{
			float bgc = bg_color[ch];
			if (floorW > 0.0f)
			{
				const float fc = 0.50f + 0.32f * floorShade;   // 어두운 칸 0.50 / 밝은 칸 0.82
				bgc += (fc - bgc) * floorW;
			}
			out_color[ch * H * W + pix_id] = C[ch] + T * bgc;
		}
		if (id_buffer != nullptr && picked_id != -1)
		{
			id_buffer[pix_id] = picked_id;//pix_id 는 픽셀인덱스 즉 몇번픽셀에 몇번가우시안인지 담기. 
		}
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	int* id_buffer)
{
	// 체커 바닥 파라미터 (setGroundRender가 매 프레임 채운다). 꺼져 있으면 커널이 값을 쓰지 않는다.
	const float* m = g_groundViewM;
	const float tx = g_groundTanFovX, ty = g_groundTanFovY;
	float t1[3], t2[3];
	groundTangentBasis(g_groundN, t1, t2);
	const float invChecker = 1.0f / fmaxf(g_groundChecker, 1e-8f);
	const float3 grCam = make_float3(g_groundCamPos[0], g_groundCamPos[1], g_groundCamPos[2]);
	const float3 grRx = make_float3(m[0] * tx, m[4] * tx, m[8] * tx);   // view R 의 0행 = 카메라 x축(world)
	const float3 grRy = make_float3(m[1] * ty, m[5] * ty, m[9] * ty);
	const float3 grRz = make_float3(m[2], m[6], m[10]);
	const float3 grN = make_float3(g_groundN[0], g_groundN[1], g_groundN[2]);
	const float3 grT1 = make_float3(t1[0] * invChecker, t1[1] * invChecker, t1[2] * invChecker);
	const float3 grT2 = make_float3(t2[0] * invChecker, t2[1] * invChecker, t2[2] * invChecker);
	const float3 grOrigin = make_float3(g_groundOrigin[0], g_groundOrigin[1], g_groundOrigin[2]);
	const float grCamH = g_groundN[0] * g_groundCamPos[0] + g_groundN[1] * g_groundCamPos[1]
		+ g_groundN[2] * g_groundCamPos[2] - g_groundHeight;
	const float grPixScale = 2.0f * ty / (float)(H > 0 ? H : 1) * invChecker;   // 세로 픽셀 각크기 / 칸 크기
	const float grInvFade2 = 1.0f / fmaxf(g_groundFadeRadius * g_groundFadeRadius, 1e-12f);

	//renderCUDA는 한 타일(tile)에 해당하는 가우시안 목록을 shared memory로 배치해서 처리
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,//타일별로 깊이 정렬된 가우시안 인덱스 리스트
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		id_buffer,
		g_groundVisible ? 1 : 0,
		grCam, grRx, grRy, grRz, grN, grT1, grT2, grOrigin,
		grCamH, grPixScale, grInvFade2);
}
bool FORWARD::ChainMail::loadGraph(
	ChainMail& cm,
	const std::vector<Pos>& cropped_pos,
	const std::vector<Edge>& cropped_edges,
	const std::vector<float>& cropped_opacity
) {

	// 1. Elements 초기화
	cm.elements.clear();
	cm.elements.resize(cropped_pos.size());
	for (int i = 0; i < cropped_pos.size(); ++i) {
		cm.elements[i].pos = cropped_pos[i];
		cm.elements[i].vel = glm::vec3(0.0f);
		cm.elements[i].invMass = 1.0f;
		cm.elements[i].density = cropped_opacity[i];     // 기본값
		cm.elements[i].time = 1e9f;
		cm.elements[i].offset = 0;
		cm.elements[i].neighborCnt = 0;
	}

	// 2. Edge Neighbor 변환 준비
	cm.cedges.clear();
	cm.cedges.reserve(cropped_edges.size());
	std::vector<std::vector<Neighbor>> tempNeigh(cropped_pos.size());
	for (const auto& e : cropped_edges) {
		const int a = e.m_vert[0];
		const int b = e.m_vert[1];
		if (a < 0 || b < 0 || a >= (int)cropped_pos.size() || b >= (int)cropped_pos.size() || a == b) {
			continue;
		}
		tempNeigh[a].emplace_back(b, e.rl, e.st);
		tempNeigh[b].emplace_back(a, e.rl, e.st);
		FORWARD::cEdge ce;
		ce.v1 = a;
		ce.v2 = b;
		ce.dist = e.rl;
		cm.cedges.push_back(ce);
	}

	// 3. Neighbor 배열 Flatten
	cm.neighbors.clear();
	int idx = 0, offset = 0;
	for (int i = 0; i < cropped_pos.size(); ++i) {
		auto& elem = cm.elements[i];
		elem.offset = offset;
		elem.neighborCnt = static_cast<int>(tempNeigh[i].size());

		for (auto& n : tempNeigh[i]) {
			cm.neighbors.push_back(n);
			idx++;
		}
		offset = idx;
	}

	// 차수 분포. 변형 그래디언트가 이웃을 전부 보고 있는지 확인하는 용도.
	{
		const size_t n = cm.elements.size();
		int minDeg = INT_MAX, maxDeg = 0;
		long long sumDeg = 0;
		int isolated = 0, degLt3 = 0;
		for (size_t i = 0; i < n; ++i) {
			const int d = cm.elements[i].neighborCnt;
			minDeg = std::min(minDeg, d);
			maxDeg = std::max(maxDeg, d);
			sumDeg += d;
			if (d == 0) ++isolated;
			if (d < 3) ++degLt3;
		}
		if (n > 0) {
			printf("[ChainMail graph] nodes=%zu edges=%zu | degree min=%d max=%d avg=%.2f\n",
				n, cm.cedges.size(), (minDeg == INT_MAX ? 0 : minDeg), maxDeg,
				(double)sumDeg / (double)n);
			printf("[ChainMail graph] isolated=%d (%.2f%%), degree<3 (F=identity)=%d (%.2f%%)\n",
				isolated, 100.0 * isolated / n, degLt3, 100.0 * degLt3 / n);
		}
	}
	return true;
}
//void FORWARD::SetChainMailGraphFromVectors(
//	const std::vector<Pos>& cropped_pos,
//	const std::vector<Edge>& cropped_edges,
//	const std::vector<float>& cropped_opacity
//) {
//	// 그래프는 CPU에서 한 번만 구성
//	/*FORWARD::g_chainmail.loadGraph(g_chainmail, cropped_pos, cropped_edges, cropped_opacity);
//	FORWARD::g_chainmail_ready = true;}*/


FORWARD::CMConstraint h_AIR(
	0.1f, 0.1f, 0.1f,   // dx, dy, dz
	0.1f, 0.1f,         // xShearY, xShearZ
	0.1f, 0.1f,         // yShearX, yShearZ
	0.1f, 0.1f          // zShearX, zShearY
);

// SKIN
FORWARD::CMConstraint h_SKIN(
	0.01f, 0.01f, 0.01f,
	0.01f, 0.01f,
	0.01f, 0.01f,
	0.01f, 0.01f
);

// BONE
FORWARD::CMConstraint h_BONE(
	0.0001f, 0.0001f, 0.0001f,
	0.0001f, 0.0001f,
	0.0001f, 0.0001f,
	0.0001f, 0.0001f
);




void FORWARD::ChainMail::resetTime() {
	for (auto& e : elements)
		e.time = 1e9f;
}


void FORWARD::ChainMail::movePointPos(int* idx, const glm::vec3& dpos, std::vector<int>& activeSet) {
	if (g_gpuCommandMode) {
		// Per-frame drag calls can be very frequent; avoid log spam here.
		enqueueGpuCommand(*idx, dpos);
		activeSet.push_back(*idx);
		return;
	}
	// Per-frame drag calls can be very frequent; avoid log spam here.
	elements[*idx].pos = elements[*idx].pos + dpos;
	elements[*idx].time = 0.0f;
	activeSet.push_back(*idx);
}
/**
 * @brief 시작점으로부터 그래프 연결성을 따라 n개의 인접 정점을 수집함
 * @param startNode 마우스 클릭 등으로 선택된 시작 가우시안 인덱스
 * @param targetCount 수집할 총 정점 개수 (예: 50개)
 * @return 수집된 인덱스들의 벡터
 */
#include <vector>
#include <queue>
#include <algorithm>
std::vector<int> FORWARD::ChainMail::collectSeedsBFS(int startNode, int targetCount) {
	std::vector<int> selectedSeeds;
	if (startNode < 0 || startNode >= elements.size()) return selectedSeeds;

	std::queue<int> q;
	std::vector<bool> visited(elements.size(), false);

	// 시작점 설정
	q.push(startNode);
	visited[startNode] = true;

	while (!q.empty() && selectedSeeds.size() < targetCount) {
		int curr = q.front();
		q.pop();

		selectedSeeds.push_back(curr);

		// 현재 노드의 이웃들을 탐색
		const Element& elem = elements[curr];
		for (int i = 0; i < elem.neighborCnt; ++i) {
			int neighborIdx = neighbors[elem.offset + i].idx;

			// 아직 방문하지 않은 이웃만 큐에 삽입
			if (neighborIdx >= 0 && !visited[neighborIdx]) {
				visited[neighborIdx] = true;
				q.push(neighborIdx);
			}

			// 목표 개수를 채우면 즉시 중단
			if (selectedSeeds.size() + q.size() > targetCount * 2) {
				// 큐가 너무 커지는 것을 방지하기 위한 조기 종료 로직 (선택 사항)
			}
		}
	}

	return selectedSeeds;
}
// 여러 점을 동시에 시작점으로 설정하는 방식
void FORWARD::ChainMail::startWaveMultiple(const std::vector<int>& seeds, glm::vec3 delta, std::vector<int>& activeSet)
{
	if (g_gpuCommandMode) {
		printf("g_gpuCommandMode movePointIdx : %d  moveDelta : %f , %f , %f\n", seeds.size(), delta.x, delta.y, delta.z);
		for (int sIdx : seeds) {
			enqueueGpuCommand(sIdx, delta);
			activeSet.push_back(sIdx);
		}
		return;
	}
	else {
		printf("cpuCommandMode movePointIdx : %d  moveDelta : %f , %f , %f\n", seeds.size(), delta.x, delta.y, delta.z);
		for (int sIdx : seeds) {
			elements[sIdx].pos += delta; // 50개 노드를 동시에 delta만큼 이동
			elements[sIdx].time = std::min(elements[sIdx].time, 0.0f);

			//elements[sIdx].time = 0.0f;
			activeSet.push_back(sIdx);
		}
	}



}
void FORWARD::ChainMail::startWavingMultiple(
	const std::vector<int>& seeds,
	glm::vec3 delta,
	std::vector<int>& activeSet)
{
	waveActive.clear();

	int N = seeds.size();

	for (int i = 0; i < N; ++i) {
		int idx = seeds[i];

		//float weight = 1.0f - (i / float(N));  // 여기
		//elements[idx].pos += delta * weight;

		elements[idx].time = 0.0f;
		activeSet.push_back(idx);
		waveActive.push_back(idx);
	}

	waveRunning = true;
}
void FORWARD::ChainMail::startWave(int seed, glm::vec3 delta, std::vector<int>& activeSet)
{
	if (g_gpuCommandMode) {
		enqueueGpuCommand(seed, delta);
		activeSet.push_back(seed);
		waveActive.clear();
		waveActive.push_back(seed);
		return;
	}
	elements[seed].pos += delta;
	elements[seed].time = 0.0f;
	activeSet.push_back(seed);
	waveActive.clear();
	waveActive.push_back(seed);
	//waveRunning = true;
}
void FORWARD::ChainMail::applyWaveOffset(
	const glm::vec3& delta,
	std::vector<int>& activeSet)
{
	if (g_gpuCommandMode) {
		for (int idx : seeds) {
			enqueueGpuCommand(idx, delta);
			activeSet.push_back(idx);
		}
		return;
	}
	for (int idx : seeds) {
		elements[idx].pos += delta;   // 위치만 이동
		activeSet.push_back(idx);     // relax 대상
	}
}


void FORWARD::ChainMail::propagateStep(const std::vector<int>& currentFrontier,std::vector<int>& nextFrontier, std::vector<int>& totalActiveSet)
{
	// static 변수 제거! (매 호출마다 상태가 초기화되어야 함)

	// 이번 단계에서 처리할 노드들에 대해 방문 표시가 필요하다면
	// ChainMail 특성상 time 비교를 하므로 별도 visited 배열이 없어도 되지만,
	// 한 프레임 내 중복 방지를 위해 로컬 visited를 쓰거나 time 체크를 철저히 해야 합니다.

	for (int idx : currentFrontier) {
		Element& elem = elements[idx];

		for (int ni = 0; ni < elem.neighborCnt; ++ni) {
			const Neighbor& neigh = neighbors[elem.offset + ni];
			int nIdx = neigh.idx;
			float dist = neigh.dist;

			Element& neighbor = elements[nIdx];

			float newTime = elem.time + propagationTime(elem, neighbor);

			// 더 빠른 경로(더 강력한 당김)가 발견되면 업데이트
			if (neighbor.time > newTime) {
				neighbor.time = newTime;

				bool moved = false;
				shiftElementPoint(neighbor, elem, dist, moved);

				if (moved) {
					// 이번에 움직였으면 다음 단계에서 얘의 이웃도 검사해야 함
					nextFrontier.push_back(nIdx);

					// Relax를 위해 전체 목록에도 추가
					totalActiveSet.push_back(nIdx);
				}
			}
		}
	}

	// 중복 제거 (선택 사항이나 성능을 위해 권장)
	if (!nextFrontier.empty()) {
		std::sort(nextFrontier.begin(), nextFrontier.end());
		nextFrontier.erase(std::unique(nextFrontier.begin(), nextFrontier.end()), nextFrontier.end());
	}
}

void FORWARD::ChainMail::propagate(std::vector<int>& activeSet)
{
	size_t N = elements.size();

	// --- persistent wave state ---
	static std::vector<int> active;
	static bool initialized = false;

	// 첫 호출 시 seed 초기화
	if (!initialized) {
		active.clear();
		for (size_t i = 0; i < N; ++i)
			if (elements[i].time == 0.0f)
				active.push_back((int)i);
		initialized = true;
	}

	if (active.empty()) {
		initialized = false; // wave 종료 다음 클릭 때 재시작
		return;
	}

	std::vector<int> nextActive;
	std::vector<bool> visited(N, false);

	for (int idx : active)
		visited[idx] = true;

	//  여기서 1 iteration만 수행
	for (int idx : active) {
		Element& elem = elements[idx];

		for (int ni = 0; ni < elem.neighborCnt; ++ni) {
			const Neighbor& neigh = neighbors[elem.offset + ni];
			int nIdx = neigh.idx;
			float dist = neigh.dist;

			Element& neighbor = elements[nIdx];

			float newTime = elem.time + propagationTime(elem, neighbor);

			if (neighbor.time > newTime) {
				neighbor.time = newTime;

				bool moved = false;
				shiftElementPoint(neighbor, elem, dist, moved);

				if (moved) {
					activeSet.push_back(nIdx);
					//printf("propagate activeSet size (실제 움직임):%d\n ", activeSet.size());
				}

				if (!visited[nIdx]) {
					nextActive.push_back(nIdx);
					visited[nIdx] = true;
				}
			}

		}
	}

	active = std::move(nextActive);


}


// //변화(전파) wave가 실제로 도달한 정점 인덱스만 activeSet에 기록 (relax에 활용)
//void FORWARD::ChainMail::propagate(std::vector<int>& activeSet) {
//	size_t N = elements.size();//전체정점 개수
//	std::vector<int> active;// 활성화 된 정점들
//	//activeSet.clear();//결과 배열
//	std::vector<bool> propagated(N, false);  // relax용: propagate로 전파된(방문된) 정점 여부
//	std::vector<bool> movedFlag(N, false);     // 실제 위치 이동이 발생한 정점 기록
//
//	// 초기 활성화: 타임스탬프 0인 정점
//	for (size_t i = 0; i < N; ++i)
//		if (elements[i].time == 0.0f)//초기에 활성화된(변형의 시작점찾기) 타임스탬스0인점
//			active.push_back(static_cast<int>(i));//활성화 목록에 추가
//
//	int iter = 0;
//	// propagation 반복
//	while (!active.empty()) {//활성화 목록에서 모두 처리가끝나면 전파끝
//		std::vector<int> nextActive;//다음 iter 에서 활성화 할 정점들
//		std::vector<bool> visited(N, false); // 같은 iter 에서 중복 생성 방지
//
//		for (int idx : active)
//			visited[idx] = true;
//
//		// #pragma omp parallel for (병렬화시 활성화)
//		for (size_t aidx = 0; aidx < active.size(); ++aidx) {
//			int idx = active[aidx];
//			Element& elem = elements[idx];
//			int off = elem.offset, nCnt = elem.neighborCnt;
//
//			for (int ni = 0; ni < nCnt; ++ni) {
//				const Neighbor& neigh = neighbors[off + ni];
//				int nIdx = neigh.idx;
//				float dist = neigh.dist;
//				//float st = neigh.st;
//
//				Element& neighbor = elements[nIdx];
//
//				// propagation time, 반드시 '내 time + link시간'!
//				float newTime = elem.time + propagationTime(elem, neighbor);
//
//				// 이웃이 현재보다 더 빠른 경로로 도달하면 update
//				if (neighbor.time > newTime) {
//					neighbor.time = newTime;
//
//					bool moved = false;
//					shiftElementPoint(neighbor, elem, dist, moved); // neighbor 이동
//					propagated[nIdx] = true;   // t와 pos 모두 변화
//
//					// activeSet 등록(실제 변화 발생)
//					if (moved) {
//						movedFlag[nIdx] = true;
//					}                    // 활성화 리스트(중복 방지)
//					if (!visited[nIdx]) {
//						nextActive.push_back(nIdx);
//						visited[nIdx] = true;
//					}
//				}
//			}
//		}
//		active = std::move(nextActive);
//		iter++;
//		//std::cout << "Iteration " << iter << ", Active Count: " << active.size() << std::endl;
//	}
//
//	// propagate wave가 실제로 도달해 변화가 일어난 모든 정점의 index를 activeSet에 기록
//	for (size_t i = 0; i < N; ++i)
//		if (propagated[i] && movedFlag[i])
//			activeSet.push_back(static_cast<int>(i));
//	printf("propagate activeSet size (실제 움직임):%d\n ", activeSet.size());
//	//std::cout << "propagate activeSet size (실제 움직임): " << activeSet.size() << std::endl;
//
//}
float FORWARD::ChainMail::propagationTime(const Element& e, const Element& neighbor) {
	// density가 0~1 범위일 때 예시
	float et, nt;
	if (e.density < AIR)        et = 1.0f;        // soft (air)
	else if (e.density < SKIN)   et = 0.3f;        // mid (skin)
	else if (e.density < BONE)  et = 0.05f;       // stiff (bone 등)
	else et = 0.005;

	if (neighbor.density < AIR)      nt = 1.0f;
	else if (neighbor.density < SKIN) nt = 0.3f;
	else if (neighbor.density < BONE) nt = 0.05f;       // stiff (bone 등)
	else nt = 0.005;
	return (et + nt) * 0.5f;

	// 또는 아주 단순하게
	// return 1.0f;   // wave 속도 고정 (실제 변화는 제약값에서 발생)
}
FORWARD::CMConstraint FORWARD::ChainMail::getConstraint(float density) {
	// 0(skin/air) ~ 1(bone)
	float axisC;   // 축 방향 강성
	float shearC;  // 전단 방향 강성

	if (density < FORWARD::AIR) {           // Air
		axisC =  0.3f;
		shearC = 0.3f;
	}
	else if (density < FORWARD::SKIN) {     // Skin
		axisC = 0.06f;
		shearC = 0.06f;// 0.06f;
	}
	else {                         // Bone
		axisC =  0.04f;
		shearC =  0.04f;
	}

	return FORWARD::CMConstraint(
		axisC, axisC, axisC,   // dx, dy, dz
		shearC, shearC,        // xShearY, xShearZ
		shearC, shearC,        // yShearX, yShearZ
		shearC, shearC         // zShearX, zShearY
	);
}


void FORWARD::ChainMail::shiftElementPoint(Element& elem, const Element& n, float targDist,  bool& moved) {
	// 간단화: density별로 constraint 하드코딩 적용(실제값은 현업코드 참조)
	CMConstraint nConstraint = getConstraint(n.density);
	// st=1이면 매우 쫀쫀(허용오차 작게), st=0이면 느슨(허용오차 크게)


	//glm::vec3 dir = n.pos - elem.pos;
	//float len = dir.length();
	//glm::vec3 nDir = norm(dir);// (len > 1e-6f) ? glm::normalize(dir) : glm::vec3();
	//
	//Vec3 dir(n.pos.x - elem.pos.x, n.pos.y - elem.pos.y, n.pos.z - elem.pos.z);
	glm::vec3 Dir = n.pos - elem.pos;
	Vec3 dir(Dir.x, Dir.y, Dir.z);
	float len = dir.length();
	glm::vec3 nDir = glm::normalize(Dir);

	float delta = 0.0f;
	float alpha = 1.0f; // 0 < alpha <= 1.0, 낮을수록 부드럽게
	if (len < targDist - nConstraint.dx) {
		// 너무 가까움 멀어져야 함 -nDir 방향
		delta = (targDist - nConstraint.dx) - len;
		elem.pos = elem.pos - nDir * delta;
		moved = true;
	}
	else if (len > targDist + nConstraint.dx) {
		// 너무 멀음  가까워져야 함 +nDir 방향
		delta = len - (targDist + nConstraint.dx);
		elem.pos = elem.pos + nDir * delta;
		moved = true;
	}
	else {
		moved = false;
	}


	//if (len < targDist - nConstraint.dx) {//제약 범위보다 가깝거나
	//    delta = (targDist - nConstraint.dx) - len;//제약범위내에서 가능한 가깝게 설정
	//}
	//else if (len > targDist + nConstraint.dx) {//제약범위보다 멀면
	//    delta = (targDist + nConstraint.dx) - len;//거리 제약범위 내에서 최대거리로 설정
	//}

	//// 한 번에 다 보내지 않고 조금씩 따라오게!
	//if (std::abs(delta) > 1e-6f) {
	//    float alpha = 1.0f; // 0 < alpha <= 1.0, 낮을수록 부드럽게
	//    elem.pos = elem.pos + nDir * (delta * alpha);
	//    moved = true;
	//}
	//else {
	//    moved = false;
	//}
}


const FORWARD::Element& FORWARD::ChainMail::getElement(int i) const {
	return elements[i];
}

FORWARD::Element& FORWARD::ChainMail::getElement(int i) {
	return elements[i];
}
const FORWARD::Neighbor& FORWARD::ChainMail::getNeighbor(int i) const {
	return neighbors[i];
}

FORWARD::Neighbor& FORWARD::ChainMail::getNeighbor(int i) {
	return neighbors[i];
}

const std::vector<FORWARD::cEdge>& FORWARD::ChainMail::getEdges() const
{
	return cedges;
}
void FORWARD::ChainMail::B_relax(const std::vector<int>& activeSet) {
	std::vector<glm::vec3> newPos(elements.size());

	float a = 0.01f;
	float b = 0.01f;
	for (int idx : activeSet) {
		Element& e = elements[idx];

		// 자기 밀도에 따른 제약값 (getConstraint 대체)
		CMConstraint eConstraint;
		if (e.density < AIR) {          // AIR
			eConstraint.dx = 0.8f;  eConstraint.dy = 0.8f;  eConstraint.dz = 0.8f;
			eConstraint.xShearY = eConstraint.xShearZ = 0.8f;
			eConstraint.yShearX = eConstraint.yShearZ = 0.8f;
			eConstraint.zShearX = eConstraint.zShearY = 0.8f;
		}
		else if (e.density < SKIN) {    // SKIN
			eConstraint.dx = a;  eConstraint.dy = a;  eConstraint.dz = a;
			eConstraint.xShearY = eConstraint.xShearZ = a;
			eConstraint.yShearX = eConstraint.yShearZ = a;
			eConstraint.zShearX = eConstraint.zShearY = a;
		}
		else {                          // BONE
			eConstraint.dx = b;  eConstraint.dy = b;  eConstraint.dz = b;
			eConstraint.xShearY = eConstraint.xShearZ = b;
			eConstraint.yShearX = eConstraint.yShearZ = b;
			eConstraint.zShearX = eConstraint.zShearY = b;
		}

		glm::vec3 sumPos(0, 0, 0);
		glm::vec3 totalWeight(0, 0, 0);
		int nCnt = 0;

		// 6방향 이웃 처리
		for (int j = 0; j < e.neighborCnt; ++j) {
			const Neighbor& n = neighbors[e.offset + j];
			const Element& nb = elements[n.idx];
			float targetDist = n.dist;

			// 이웃 밀도에 따른 제약값
			CMConstraint nConstraint;
			if (nb.density < AIR) {
				nConstraint.dx = 0.8f;  nConstraint.dy = 0.8f;  nConstraint.dz = 0.8f;
				nConstraint.xShearY = nConstraint.xShearZ = 0.8f;
				nConstraint.yShearX = nConstraint.yShearZ = 0.8f;
				nConstraint.zShearX = nConstraint.zShearY = 0.8f;
			}
			else if (nb.density < SKIN) {
				nConstraint.dx = a;  nConstraint.dy = a;  nConstraint.dz = a;
				nConstraint.xShearY = nConstraint.xShearZ = a;
				nConstraint.yShearX = nConstraint.yShearZ = a;
				nConstraint.zShearX = nConstraint.zShearY = a;
			}
			else {
				nConstraint.dx = b;  nConstraint.dy = b;  nConstraint.dz = b;
				nConstraint.xShearY = nConstraint.xShearZ = b;
				nConstraint.yShearX = nConstraint.yShearZ = b;
				nConstraint.zShearX = nConstraint.zShearY = b;
			}

			// 방향별 가중치 계산 (간단화)
			float wX = 1.0f / ((eConstraint.dx + nConstraint.dx+targetDist) * 0.5f + 1e-6f);
			float wY = 1.0f / ((eConstraint.dy + nConstraint.dy+targetDist) * 0.5f + 1e-6f);
			float wZ = 1.0f / ((eConstraint.dz + nConstraint.dz+targetDist) * 0.5f + 1e-6f);

			sumPos.x += nb.pos.x * wX;
			sumPos.y += nb.pos.y * wY;
			sumPos.z += nb.pos.z * wZ;

			totalWeight.x += wX;
			totalWeight.y += wY;
			totalWeight.z += wZ;

			nCnt++;
		}

		if (nCnt > 0) {
			glm::vec3 newE;
			newE.x = sumPos.x / totalWeight.x;
			newE.y = sumPos.y / totalWeight.y;
			newE.z = sumPos.z / totalWeight.z;
			newPos[idx] = newE;
		}
		else {
			newPos[idx] = e.pos;
		}
	}

	// 위치 갱신
	for (int idx : activeSet) {
		elements[idx].pos = newPos[idx];
	}
}

void FORWARD::ChainMail::Stabilize(const std::vector<int>& activeSet) {
	// 거리 제약 기반 위치 보정
	for (int idx : activeSet) {
		Element& e = elements[idx];

		for (int j = 0; j < e.neighborCnt; ++j) {
			Neighbor& n = neighbors[e.offset + j];
			Element& nb = elements[n.idx];

			glm::vec3 delta = e.pos - nb.pos;
			float dist = delta.length();
			if (dist < 1e-6f) continue;

			float diff = dist - n.dist; // targetDist와 현재 거리 차이
			// correction 비율 (0.5 → 양쪽 절반씩 이동)
			glm::vec3 correction = (diff / dist) * 0.5f * delta;

			// 밀도/제약 고려해서 움직임 크기 제한 가능
			e.pos -= correction;
			nb.pos += correction;
		}
	}
}
void FORWARD::ChainMail::relax2(const std::vector<int>& activeSet) {
	// [변경 1] newPos 벡터 제거 (메모리 절약 + 속도 향상)
	// std::vector<glm::vec3> newPos(elements.size()); 

	float stiffness = 0.9f; // 원하시는 대로 유지

	for (int idx : activeSet) {
		Element& e = elements[idx];
		CMConstraint eConstraint = getConstraint(e.density);

		glm::vec3 accumCorrection(0.0f);
		int corrCount = 0;
		float sumTargetDist = 0.0f;
		int distCount = 0;

		for (int j = 0; j < e.neighborCnt; ++j) {
			const Neighbor& n = neighbors[e.offset + j];

			// [핵심] n.idx의 위치를 가져올 때, 앞서 계산된 이웃이라면 
			// 이미 보정된 '최신 위치'를 가져오게 되어 전파 속도가 2배 이상 빨라집니다.
			const Element& nb = elements[n.idx];
			CMConstraint nConstraint = getConstraint(nb.density);

			float targDist = n.dist;
			float tol = 0.1f * (eConstraint.dx + nConstraint.dx) * 0.02f;

			glm::vec3 dir = nb.pos - e.pos;
			float len = glm::length(dir);
			if (len < 1e-6f) continue;

			glm::vec3 nDir = dir / len;

			if (len < targDist - tol) { // 너무 가까움 (압축)
				float delta = (targDist - tol) - len;
				accumCorrection -= nDir * delta;
				corrCount++;
			}
			else if (len > targDist + tol) { // 너무 멈 (인장)
				float delta = len - (targDist + tol);
				accumCorrection += nDir * delta;
				corrCount++;
			}

			sumTargetDist += targDist;
			distCount++;
		}

		if (corrCount > 0) {
			glm::vec3 avgCorr = accumCorrection / float(corrCount);

			// --- [변경 2] 속도 제한(Limit) 완화 ---
			// 기존 1.5배는 돌아오려는 움직임을 너무 강하게 브레이크 잡습니다.
			// 이걸 4.0배 정도로 늘려주면 안정성은 유지하되 '확' 돌아옵니다.
			float maxStep = 0.8f;
			if (distCount > 0) {
				float avgTarget = sumTargetDist / float(distCount);
				const float maxStepFactor = 8.0f; // 기존 1.5f -> 4.0f로 증가 (이게 속도의 핵심)
				maxStep = maxStepFactor * avgTarget;
			}

			// 너무 큰 보정은 잘라낸다 (폭발 방지용 안전장치)
			float corrLen = glm::length(avgCorr);
			if (corrLen > 1e-6f) {
				if (corrLen > maxStep) {
					avgCorr = avgCorr * (maxStep / corrLen);
				}

				// [변경 3] 즉시 적용 (Gauss-Seidel)
				// newPos에 넣지 않고 내 위치를 바로 바꿉니다.
				e.pos += stiffness * avgCorr;
			}
		}
	}

	// [변경 4] 마지막 복사 루프 제거 (위에서 이미 적용함)
	// for (int idx : activeSet) elements[idx].pos = newPos[idx];
}
void FORWARD::ChainMail::relax(const std::vector<int>& activeSet) {
	std::vector<glm::vec3> newPos(elements.size());
	float stiffness = 0.8f; // 권장 범위 0.2 ~ 0.5

	for (int idx : activeSet) {
		Element& e = elements[idx];
		CMConstraint eConstraint = getConstraint(e.density);

		glm::vec3 accumCorrection(0.0f);
		int corrCount = 0;
		float sumTargetDist = 0.0f;
		int distCount = 0;

		for (int j = 0; j < e.neighborCnt; ++j) {
			const Neighbor& n = neighbors[e.offset + j];
			const Element& nb = elements[n.idx];
			CMConstraint nConstraint = getConstraint(nb.density);

			float targDist = n.dist;
			float tol = 0.1*(eConstraint.dx + nConstraint.dx) * 0.02f;

			glm::vec3 dir = nb.pos - e.pos;
			float len = glm::length(dir);
			if (len < 1e-6f) continue;

			glm::vec3 nDir = dir / len;

			if (len < targDist - tol) {//너무 가까울시 이웃반대방향으로 초과된범위만큼 멀어짐
				float delta = (targDist - tol) - len;
				accumCorrection -= nDir * delta;
				corrCount++;
			}
			else if (len > targDist + tol) {//너무 멀때 이웃방향으로 제약초과범위만큼 가까워짐
				float delta = len - (targDist + tol);
				accumCorrection += nDir * delta;
				corrCount++;
			}

			// 로컬 평균 목표거리 수집
			sumTargetDist += targDist;
			distCount++;
		}

		if (corrCount > 0) {
			glm::vec3 avgCorr = accumCorrection / float(corrCount);

			// --- 로컬 maxStep 계산 ---
			float maxStep = 0.6f; // 기본 fallback
			if (distCount > 0) {
				float avgTarget = sumTargetDist / float(distCount);
				const float maxStepFactor = 1.5f; // avgTarget의 몇 배를 최대 스텝으로 허용할지
				maxStep = maxStepFactor * avgTarget;
			}

			// 너무 큰 보정은 잘라낸다
			float corrLen = glm::length(avgCorr);
			if (corrLen > 1e-6f && corrLen > maxStep) {
				avgCorr = avgCorr * (maxStep / corrLen);
			}

			glm::vec3 corrected = e.pos + stiffness * avgCorr;
			newPos[idx] = corrected;
		}
		else {
			newPos[idx] = e.pos;
		}
	}

	for (int idx : activeSet) elements[idx].pos = newPos[idx];
}


size_t FORWARD::ChainMail::numElements() const

{
	return elements.size();
}
// ===============================
// GPU persistent chainmail state
// ===============================
static float3* d_pos_curr = nullptr;
static float3* d_pos_next = nullptr;
static float3* d_pos_xpbd_tmp = nullptr;
static float3* d_pos_rest = nullptr;
static float3* d_vel = nullptr;
static float* d_density = nullptr;
static float* d_invMass = nullptr;
static float* d_time_curr = nullptr;
static float* d_time_next = nullptr;
static int* d_offset = nullptr;
static int* d_nbrCount = nullptr;
static int* d_nbrIdx = nullptr;
static float* d_nbrDist = nullptr;
static float* d_nbrStiff = nullptr;
// Per directed neighbor edge (same indexing as d_nbrIdx):
// - d_rest_cos[e]: legacy angle metric (rest dot = v1·v2) between edge e and its ordered umbrella successor.
// - d_pair_next_idx[e]: successor neighbor index used for angle constraint pairing.
static float* d_rest_cos = nullptr;
static int* d_pair_next_idx = nullptr;
// d_rest_cos stores rest cosine for each directed umbrella edge pair.
static float3* d_angle_dp_sum = nullptr;
static int* d_angle_dp_count = nullptr;
static float* d_lambda_curr = nullptr;
static float* d_lambda_next = nullptr;
// ===============================
// Volume Gaussian (부피 가우시안) — 클러스터 공분산으로 사면체를 대신한다.
// Sigma_rest: rest 클러스터 공분산 (대칭 6원소: xx, yy, zz, xy, xz, yz)
// detSigmaRest: det(Sigma_rest). J = sqrt(det Sigma_cur / det Sigma_rest) = |det F|
// V_rest: (4/3)*pi*sqrt(det Sigma_rest) — 사면체의 rest 부피를 대신한다.
// matType: 0=volume, 1=surface, 2=fiber. 고유값 이방성으로 자동 분류.
// ===============================
static float* d_Sigma_rest = nullptr;   // [N*6]
static float* d_detSigmaRest = nullptr; // [N]
static float* d_V_rest = nullptr;       // [N]
static float* d_volRestLen = nullptr;   // [N] 이웃 rest 평균거리 (스텝 클램프 길이 스케일)
static int* d_matType = nullptr;        // [N]
static float* d_lambda_vol = nullptr;   // [N] 부피 제약 Lagrange multiplier
static float* d_alpha_vol = nullptr;    // [N] compliance (나중에 FEM 학습 결과가 들어옴)
// Gaussian-cluster Stable Neo-Hookean experimental path. These buffers never
// alias the existing volume or shape-matching solver state.
static float3* d_gnhRestC = nullptr;        // [N] rest centroid per valid leader
static float* d_gnhRestSinv = nullptr;      // [N*6] inverse rest covariance
static unsigned char* d_gnhValid = nullptr; // [N] full-rank volume cluster
static float* d_lambda_gnhD = nullptr;      // [N] distortional multiplier
static float* d_lambda_gnhH = nullptr;      // [N] hydrostatic multiplier
static float* d_gnhClusterF = nullptr;      // [N*9] column-major best-fit F
static float* d_gnhClusterCoefD = nullptr;  // [N] dLambda_D/(n*||F||_F)
static float* d_gnhClusterCoefH = nullptr;  // [N] dLambda_H/n
// 거리 GS: 무향 간선(i<j)을 색 순서로 정렬해 둔다. 같은 색 안에서는 점을 공유하지 않으므로
// 한 색을 한 커널로 제자리(in-place) 갱신해도 경쟁이 없다.
static int2* d_gsEdge = nullptr;    // [E] (i, j)
static float* d_gsRest = nullptr;   // [E] rest 길이 (양방향이 있으면 평균)
static float* d_gsStiff = nullptr;  // [E] 강성 (양방향이 있으면 평균)
static float* d_gsLambda = nullptr; // [E] Lagrange multiplier (substep 시작 시 0)
static std::vector<int> g_gsColorOffset; // [색+1] 색별 간선 구간
static int g_gsNumEdges = 0;
static int g_gsOneWayEdges = 0;          // 한쪽 방향만 있던 간선 수 (Jacobi 에선 한쪽 점만 움직였다)
// 부피 GS: 클러스터(리더)를 색 순서로 정렬. 같은 색 = 멤버(리더 자신 포함)를 공유하지 않음.
static int* d_volGSOrder = nullptr;          // [리더 수]
static std::vector<int> g_volGSColorOffset;  // [색+1]
static int g_volGSLowerBound = 0;            // 색 수 하한 = 한 가우시안을 공유하는 클러스터 수의 최댓값
static bool g_volGSSkipped = false;          // 겹침이 너무 조밀해 컬러링을 건너뜀
// 볼륨 제약 전용 k-ring 클러스터 CSR (물리 그래프 CSR과 별개)
static int* d_volOffset = nullptr; // [N]
static int* d_volCount = nullptr;  // [N]
static int* d_volIdx = nullptr;    // [멤버 총합]
// 볼륨 제약 2패스(산란→평균 적용)용 누적 버퍼
static float3* d_vol_dp_sum = nullptr; // [N]
static int* d_vol_dp_count = nullptr;  // [N]
// ── gather 모드용 (scatter와 수학적으로 동일, 실행 전략만 다름) ──
// 전치 CSR: "노드 j를 멤버로 포함하는 클러스터들의 목록" (자기 클러스터 포함)
static int* d_volRevOffset = nullptr;  // [N]
static int* d_volRevCount = nullptr;   // [N]
static int* d_volRevIdx = nullptr;     // [멤버 총합 + N]  (self 엣지 포함)
// Kernel A가 저장하고 Kernel B가 읽는 클러스터 파라미터
static float3* d_volClusterC = nullptr;    // [N]   현재 무게중심 c_i
static float* d_volClusterSinv = nullptr;  // [N*6] Σ_i⁻¹ (대칭 6원소)
static float* d_volClusterCoef = nullptr;  // [N]   dLam_i·(J_i/n_i). 0이면 이 클러스터는 이번 반복에 무효
// Region Balloon uses one picked BFS region as a single covariance ellipsoid.
// The topology is still the same static graph; only the picked region index list changes.
static int* d_regionBalloonIdx = nullptr;     // [N max]
static float* d_regionBalloonStats = nullptr; // [9] sum xyz + covariance-sum xx yy zz xy xz yz
static int g_regionBalloonCount = 0;
static float g_regionBalloonDetRest = 0.0f;
static float g_regionBalloonRestScale = 0.0f;
static int g_regionBalloonAnchor = -1;
static int* d_active_map = nullptr;
static int* d_next_map = nullptr;
static int* d_active_count = nullptr;
static int* d_best_from = nullptr;
// Latest deformed positions used for rendering (xyz packed, 3*P floats).
// Updated every frame in preprocess().
static float* g_latest_deformed_xyz = nullptr;
static int g_latest_deformed_count = 0;
// Latest screen-space projection buffers used by render path.
static float2* g_latest_means2d = nullptr;
static int* g_latest_radii = nullptr;
static int g_latest_project_count = 0;
static int g_latest_render_w = 0;
static int g_latest_render_h = 0;
static int cm_num_elements = 0;
static int cm_num_neighbors = 0;
static bool chainmailInit = false;

void FORWARD::setPickingParams(int hops)
{
	bfsHops = (hops < 0) ? 0 : hops;
	
}

void FORWARD::getPickingParams(int* hops)
{
	if (hops) *hops = bfsHops;
	
}

bool FORWARD::copyCurrentDeformedPositions(float* outXYZ, int pointCount)
{
	if (!outXYZ || pointCount <= 0 || !g_latest_deformed_xyz || g_latest_deformed_count <= 0) {
		return false;
	}
	const int copyCount = (pointCount < g_latest_deformed_count) ? pointCount : g_latest_deformed_count;
	cudaMemcpy(outXYZ, g_latest_deformed_xyz, sizeof(float) * 3 * copyCount, cudaMemcpyDeviceToHost);
	return true;
}

bool FORWARD::copyCurrentScreenProjection(
	float* outXY,
	int* outRadii,
	int pointCount,
	int* outRenderW,
	int* outRenderH)
{
	if (!outXY || pointCount <= 0 || !g_latest_means2d || g_latest_project_count <= 0) {
		return false;
	}
	const int copyCount = (pointCount < g_latest_project_count) ? pointCount : g_latest_project_count;
	cudaMemcpy(outXY, g_latest_means2d, sizeof(float2) * copyCount, cudaMemcpyDeviceToHost);
	if (outRadii && g_latest_radii) {
		cudaMemcpy(outRadii, g_latest_radii, sizeof(int) * copyCount, cudaMemcpyDeviceToHost);
	}
	if (outRenderW) *outRenderW = g_latest_render_w;
	if (outRenderH) *outRenderH = g_latest_render_h;
	return true;
}

void FORWARD::setChainmailActiveMapEnabled(bool enabled)
{
	g_useActiveMap = enabled;
	g_activeMapReset = true;
}

float FORWARD::getChainmailActiveRatio()
{
	return g_lastActiveRatio;
}

int FORWARD::getChainmailActiveCount()
{
	return g_lastActiveCount;
}

void FORWARD::setPhysicsMode(int mode)
{
	g_physicsMode = (mode == 1) ? 1 : 0;
}

int FORWARD::getPhysicsMode()
{
	return g_physicsMode;
}

void FORWARD::setGpuChainmailMode(int mode)
{
	g_GpuChainmailMode = (mode == 1) ? 1 : 0;
	printf("g_GpuChainmailMode : %d \n", g_GpuChainmailMode);
}

int FORWARD::getGpuChainmailMode()
{
	return g_GpuChainmailMode;
}

void FORWARD::setXPBDDistanceConstraintEnabled(bool enabled)
{
	g_useXPBDDistanceConstraint = enabled;
}

bool FORWARD::getXPBDDistanceConstraintEnabled()
{
	return g_useXPBDDistanceConstraint;
}

void FORWARD::setXPBDShapeMatchingEnabled(bool enabled)
{
	g_useXPBDShapeMatching = enabled;
}

bool FORWARD::getXPBDShapeMatchingEnabled()
{
	return g_useXPBDShapeMatching;
}

void FORWARD::setXPBDShapeRobustPolar(bool enabled)
{
	g_xpbdShapeRobustPolar = enabled;
}

bool FORWARD::getXPBDShapeRobustPolar()
{
	return g_xpbdShapeRobustPolar;
}

void FORWARD::setXPBDAngleConstraintEnabled(bool enabled)
{
	g_useXPBDAngleConstraint = enabled;
}

bool FORWARD::getXPBDAngleConstraintEnabled()
{
	return g_useXPBDAngleConstraint;
}

void FORWARD::setVolumeConstraintEnabled(bool enabled)
{
	g_useVolumeConstraint = enabled;
	printf("[VolumeGaussian] constraint %s\n", enabled ? "ON" : "OFF");
}

bool FORWARD::getVolumeConstraintEnabled()
{
	return g_useVolumeConstraint;
}

void FORWARD::setGaussianNHEnabled(bool enabled)
{
	g_useGaussianNH = enabled;
	printf("[GaussianNH] experimental cluster hyperelastic solver %s\n", enabled ? "ON" : "OFF");
}

bool FORWARD::getGaussianNHEnabled()
{
	return g_useGaussianNH;
}

void FORWARD::setDistanceGSEnabled(bool enabled)
{
	g_useDistanceGS = enabled;
	printf("[DistanceGS] distance pass %s (%d edges, %d colors)\n",
		enabled ? "Gauss-Seidel" : "Jacobi", g_gsNumEdges,
		g_gsColorOffset.empty() ? 0 : int(g_gsColorOffset.size()) - 1);
}

bool FORWARD::getDistanceGSEnabled()
{
	return g_useDistanceGS;
}

void FORWARD::getDistanceGSStats(int* numEdges, int* numColors, int* oneWayEdges)
{
	if (numEdges) *numEdges = g_gsNumEdges;
	if (numColors) *numColors = g_gsColorOffset.empty() ? 0 : int(g_gsColorOffset.size()) - 1;
	if (oneWayEdges) *oneWayEdges = g_gsOneWayEdges;
}

void FORWARD::setVolumeGSEnabled(bool enabled)
{
	g_useVolumeGS = enabled;
	printf("[VolumeGS] volume pass %s (%d colors%s)\n", enabled ? "Gauss-Seidel" : "Jacobi gather",
		g_volGSColorOffset.empty() ? 0 : int(g_volGSColorOffset.size()) - 1,
		g_volGSSkipped ? ", coloring skipped - falls back to Jacobi" : "");
}

bool FORWARD::getVolumeGSEnabled()
{
	return g_useVolumeGS;
}

void FORWARD::setSolverTiming(bool enabled)
{
	g_physTiming = enabled;
	g_physTimingSum = 0.0;
	g_physTimingCount = 0;
}

void FORWARD::getSolverTiming(float* avgMs, int* samples, bool reset)
{
	if (avgMs) *avgMs = g_physTimingCount > 0 ? (float)(g_physTimingSum / g_physTimingCount) : 0.0f;
	if (samples) *samples = g_physTimingCount;
	if (reset) { g_physTimingSum = 0.0; g_physTimingCount = 0; }
}

void FORWARD::getVolumeGSStats(int* numClusters, int* numColors, int* lowerBound, bool* skipped)
{
	if (numClusters) *numClusters = g_volGSColorOffset.empty() ? 0 : g_volGSColorOffset.back();
	if (numColors) *numColors = g_volGSColorOffset.empty() ? 0 : int(g_volGSColorOffset.size()) - 1;
	if (lowerBound) *lowerBound = g_volGSLowerBound;
	if (skipped) *skipped = g_volGSSkipped;
}

void FORWARD::setGaussianNHMaterial(float young, float poisson, float complianceScale)
{
	g_gnhYoung = fmaxf(young, 1.0f);
	g_gnhPoisson = fminf(fmaxf(poisson, 1.0e-4f), 0.49f);
	g_gnhComplianceScale = fmaxf(complianceScale, 1.0e-8f);
}

void FORWARD::getGaussianNHMaterial(float* young, float* poisson, float* complianceScale)
{
	if (young) *young = g_gnhYoung;
	if (poisson) *poisson = g_gnhPoisson;
	if (complianceScale) *complianceScale = g_gnhComplianceScale;
}

void FORWARD::setVolumeParams(float compliance, float anisoThreshold)
{
	g_volCompliance = compliance;
	g_volAnisoThreshold = anisoThreshold;
}

void FORWARD::getVolumeParams(float* compliance, float* anisoThreshold)
{
	if (compliance) *compliance = g_volCompliance;
	if (anisoThreshold) *anisoThreshold = g_volAnisoThreshold;
}

void FORWARD::getVolumeMatTypeCounts(int* volumeCnt, int* surfaceCnt, int* fiberCnt)
{
	if (volumeCnt) *volumeCnt = g_volMatCount[0];
	if (surfaceCnt) *surfaceCnt = g_volMatCount[1];
	if (fiberCnt) *fiberCnt = g_volMatCount[2];
}

// invMass 리셋 커널은 아래(4200줄대)에 정의돼 있는데, squashStop/Reset이 여기서 쓰므로 선언만 앞당긴다.
__global__ void xpbdResetInvMassKernel(int N, float* invMass, float value);
// 물성 α 채우기도 정의는 아래(fillPhysicalAlphaKernel 뒤)에 있고, 여기 API가 호출하므로 선언만 앞당긴다.
static void refreshPhysicalAlpha();

// ── 물성 앵커 (Step 3) ──────────────────────────────────
void FORWARD::setVolumePhysicalMaterial(bool enabled, float E, float nu, float cvol)
{
	g_volUsePhysicalAlpha = enabled;
	g_volMatE = fmaxf(E, 1.0f);
	g_volMatNu = fminf(fmaxf(nu, 0.0f), 0.49f);
	g_volCvol = fmaxf(cvol, 1e-6f);
	refreshPhysicalAlpha();
}

void FORWARD::getVolumePhysicalMaterial(bool* enabled, float* E, float* nu, float* cvol)
{
	if (enabled) *enabled = g_volUsePhysicalAlpha;
	if (E) *E = g_volMatE;
	if (nu) *nu = g_volMatNu;
	if (cvol) *cvol = g_volCvol;
}

// ── mixture rest (명찰/신분증에 가우시안 모양 반영) ──────
void FORWARD::setVolumeMixtureRest(bool enabled)
{
	if (g_volMixtureRest == enabled) return;
	g_volMixtureRest = enabled;
	g_volTopoDirty = true; // V_rest/matType이 바뀌므로 다음 프레임에 재계산
	printf("[VolumeGaussian] mixture rest %s (V_rest/matType에 가우시안 모양 반영; 저울 J는 불변)\n",
		enabled ? "ON" : "OFF");
}

bool FORWARD::getVolumeMixtureRest() { return g_volMixtureRest; }

// ── J 시각화 (Step 2) ───────────────────────────────────
void FORWARD::setVolumeJVizEnabled(bool enabled)
{
	g_volJVizEnabled = enabled;
	printf("[VolumeGaussian] J visualization %s\n", enabled ? "ON (blue J<1 / white 1 / red J>1)" : "OFF");
}
bool FORWARD::getVolumeJVizEnabled() { return g_volJVizEnabled; }
void FORWARD::setVolumeJVizGain(float gain) { g_volJVizGain = fmaxf(0.1f, gain); }
float FORWARD::getVolumeJVizGain() { return g_volJVizGain; }

void FORWARD::setVolumeMatVizEnabled(bool enabled)
{
	g_volMatVizEnabled = enabled;
	printf("[VolumeGaussian] matType viz %s (volume=회색 / surface=파랑 / fiber=빨강)\n", enabled ? "ON" : "OFF");
}
bool FORWARD::getVolumeMatVizEnabled() { return g_volMatVizEnabled; }

// ── scatter / gather 실행 전략 ──────────────────────────
void FORWARD::setVolumeGatherMode(int mode)
{
	g_volGatherMode = (mode != 0) ? 1 : 0;
	printf("[VolumeGaussian] execution mode: %s\n",
		g_volGatherMode ? "GATHER (no atomics, deterministic)" : "SCATTER (atomicAdd)");
}

int FORWARD::getVolumeGatherMode()
{
	return g_volGatherMode;
}

// ── k-ring 클러스터 반경 ────────────────────────────────
void FORWARD::setVolumeRingK(int k, int maxMembers)
{
	k = std::max(1, std::min(8, k));
	// 상한을 HARD_CAP과 같게 둔다 — cap ≥ (k홉 공 크기)면 서브샘플이 아예 일어나지 않아
	// '전수 계산'이 된다(k=3 최대 938, k=6 최대 6,914). 그게 정확도상 원하는 상태다.
	maxMembers = std::max(8, std::min(8192, maxMembers));
	g_volRingK = k;
	g_volMaxMembers = maxMembers;
	g_volTopoDirty = true; // 다음 프레임 ensureChainmailGPU 경유로 리빌드
	printf("[VolumeGaussian] k-ring change requested: k=%d, cap=%d (rebuild next frame)\n", k, maxMembers);
}

void FORWARD::getVolumeRingK(int* k, int* maxMembers)
{
	if (k) *k = g_volRingK;
	if (maxMembers) *maxMembers = g_volMaxMembers;
}

// ── 반장 희소화 + 반원 선발 ─────────────────────────────
void FORWARD::setVolumeLeaderParams(int minHop, int memberSelect)
{
	minHop = std::max(1, std::min(8, minHop));
	// 0=stride, 1=farthest-point, 2=hop-stratified FPS.
	// (과거에 `(x != 0) ? 1 : 0` 으로 접어버려서 2를 고르면 1이 되고 새 모드가 도달 불가였다)
	memberSelect = std::max(0, std::min(2, memberSelect));
	if (g_volLeaderMinHop == minHop && g_volMemberSelect == memberSelect) return;
	g_volLeaderMinHop = minHop;
	g_volMemberSelect = memberSelect;
	g_volTopoDirty = true;
	printf("[VolumeGaussian] leader change: minHop=%d, member-select=%s (rebuild next frame)\n",
		minHop, (memberSelect == 2) ? "hop-stratified FPS" : (memberSelect == 1) ? "farthest-point" : "stride");
}

// h_j 상한 U. 0이면 비활성(기존 동작). 켜면 리빌드가 순차로 돈다.
void FORWARD::setVolumeCoverMax(int coverMax)
{
	coverMax = (coverMax < 0) ? 0 : coverMax;
	if (g_volCoverMax == coverMax) return;
	g_volCoverMax = coverMax;
	g_volTopoDirty = true;
	if (coverMax > 0)
		printf("[VolumeGaussian] h_j 상한 U=%d (rebuild next frame). 멤버 선발에서 이미 U겹 덮인 후보를 제외한다.\n", coverMax);
	else
		printf("[VolumeGaussian] h_j 상한 해제 (rebuild next frame)\n");
}
int FORWARD::getVolumeCoverMax() { return g_volCoverMax; }

// ── Σ⁻¹ 축퇴 방어 방식 (A/B) ──────────────────────────────────────────────
// false = 고유분해 + 최소축 클램프 (기존 G4, ~600 FLOP)
// true  = 등방 정규화 Σ + eps·(trace/3)·I (~15 FLOP)
// 이 구간은 멤버 수와 무관한 '직렬 구간'이라, 클러스터당 병렬화의 Amdahl 병목이 된다.
static bool  g_volSinvIsoReg = true;   // 기본값 ON (위 g_dSinvIsoReg 주석의 실측 근거)
static float g_volSinvIsoEps = 0.06f;
static float g_volSinvIsoDetThr = 0.2f;

void FORWARD::setVolumeSinvIsoReg(bool enabled, float eps, float detThr)
{
	g_volSinvIsoReg = enabled;
	g_volSinvIsoEps = fmaxf(0.0f, eps);
	g_volSinvIsoDetThr = fmaxf(0.0f, detThr);
	const int m = enabled ? 1 : 0;
	cudaMemcpyToSymbol(g_dSinvIsoReg, &m, sizeof(int));
	cudaMemcpyToSymbol(g_dSinvIsoEps, &g_volSinvIsoEps, sizeof(float));
	cudaMemcpyToSymbol(g_dSinvIsoDetThr, &g_volSinvIsoDetThr, sizeof(float));
	printf("[VolumeGaussian] Sinv 방식 = %s (eps=%.3f, detHat 임계=%.3f)\n",
		enabled ? "등방 정규화 (~15 FLOP)" : "고유분해+클램프 (~600 FLOP)",
		g_volSinvIsoEps, g_volSinvIsoDetThr);
}

// ── 렌더 변형 F: 공분산 쿠션(젤리) 실시간 토글 setter ──
static bool  g_volCovRegHost = false;
static float g_volCovLambdaHost = 1.0f;
void FORWARD::setVolumeCovReg(bool enabled, float lambda)
{
	g_volCovRegHost = enabled;
	g_volCovLambdaHost = fmaxf(0.0f, lambda);
	const int m = enabled ? 1 : 0;
	cudaMemcpyToSymbol(g_volCovReg, &m, sizeof(int));
	cudaMemcpyToSymbol(g_volCovLambda, &g_volCovLambdaHost, sizeof(float));
	printf("[RenderDeform] 공분산 쿠션(젤리) = %s (lambda=%.3f)\n",
		enabled ? "ON: 이웃 rest 공분산" : "OFF: 중심점만(기존)", g_volCovLambdaHost);
}

void FORWARD::getVolumeSinvIsoReg(bool* enabled, float* eps, float* detThr)
{
	if (enabled) *enabled = g_volSinvIsoReg;
	if (eps) *eps = g_volSinvIsoEps;
	if (detThr) *detThr = g_volSinvIsoDetThr;
}

// ── Kernel A 실행 형태 (A/B) ──────────────────────────────────────────────
// false = 클러스터당 1스레드 (기존). 멤버 위치를 못 들고 있어 3패스 전부 전역 재읽기.
// true  = 클러스터당 1워프(32). lane당 n/32명만 맡아 레지스터에 상주 → 전역 1패스.
// 메모리 대역폭에 묶인 커널이므로 이 3→1 이 이득의 본체다.
// ── Kernel A/B 실행 형태 ───────────────────────────────────────────────────
// 0 = 자동(기본) · 1 = 스레드당 1단위 · 2 = 워프당 1단위
//
// ★ 워프가 항상 빠른 게 아니다 — 클러스터 멤버 수 n 에 달렸다. dense-model measurement (r=1):
//     k=1 (n=25) :  6.55 → 8.73 ms   ← 33% 느려짐
//     k=3 (n=64) : 20.0  → 13.5 ms   ← 1.48배 빨라짐
//     k=3 (n=256): 47.5  → 21.5 ms   ← 2.21배
//     전수 + r=2 :          5.7 ms / 175 FPS
//   n이 작으면 lane당 담당이 1명 미만이라 shuffle 오버헤드(리덕션 11개 × 5단계)가
//   실제 계산을 잡아먹고, 스레드 수도 32배로 늘어 스케줄링 비용만 커진다.
//
// 그래서 자동 모드는 리빌드 때 잰 '클러스터당 평균 멤버 수'로 고른다.
// 임계 40은 실측 교차점(25 < x < 64) 사이의 값 — 씬이 바뀌면 다시 재서 조정할 것.
// 0=자동(기존 캐시형 warp 선택), 1=스레드당, 2=캐시형 워프, 3=1-pass 모멘트 워프.
// 모드 3은 A(클러스터 solve)만 바꾸며, B(gather apply)는 기존 워프 구현을 공유한다.
static int   g_volWarpMode = 0;
static float g_volAvgMembers = 0.0f;    // 마지막 리빌드의 클러스터당 평균 멤버 수
static const float VOL_WARP_AUTO_THR = 40.0f;

// 이번 프레임에 워프 커널을 쓸지. 자동 모드면 클러스터 크기로 판단한다.
static inline bool volUseWarpKernel()
{
	if (g_volWarpMode == 2 || g_volWarpMode == 3) return true;
	if (g_volWarpMode == 1) return false;
	return g_volAvgMembers >= VOL_WARP_AUTO_THR;
}

static inline bool volUseWarpOnePassKernel()
{
	return g_volWarpMode == 3;
}

void FORWARD::setVolumeWarpMode(int mode)
{
	mode = (mode < 0) ? 0 : ((mode > 3) ? 3 : mode);
	if (g_volWarpMode == mode) return;
	g_volWarpMode = mode;
	const char* nm = (mode == 0) ? "자동" : (mode == 1) ? "스레드당 1단위" :
		(mode == 2) ? "워프당 1단위 (캐시형)" : "워프당 1단위 (1-pass 모멘트)";
	printf("[VolumeGaussian] Kernel A/B 실행 형태 = %s%s\n", nm,
		(mode == 0) ? " (평균 멤버 ≥ 40 이면 워프)" : "");
}
int FORWARD::getVolumeWarpMode() { return g_volWarpMode; }
float FORWARD::getVolumeAvgMembers() { return g_volAvgMembers; }

void FORWARD::getVolumeLeaderInfo(int* minHop, int* memberSelect, int* leaderCount)
{
	if (minHop) *minHop = g_volLeaderMinHop;
	if (memberSelect) *memberSelect = g_volMemberSelect;
	if (leaderCount) *leaderCount = g_volLeaderCount;
}

// ── J 통계 ─────────────────────────────────────────────
void FORWARD::setVolumeJStatsEnabled(bool enabled) { g_volCollectJStats = enabled; }
bool FORWARD::getVolumeJStatsEnabled() { return g_volCollectJStats; }
void FORWARD::getVolumeJStats(int* count, float* mean, float* std, float* jmin, float* jmax, float* p05, float* p95)
{
	if (count) *count = g_volJStatN;
	if (mean) *mean = g_volJMean;
	if (std) *std = g_volJStd;
	if (jmin) *jmin = g_volJMin;
	if (jmax) *jmax = g_volJMax;
	if (p05) *p05 = g_volJP05;
	if (p95) *p95 = g_volJP95;
}

// 부피 가중 J (전역 부피비). 개수 평균과 크게 다르면 "작은 클러스터가 통계를 지배" 신호.
void FORWARD::getVolumeJVolumeWeighted(float* vwMean, float* vwStd)
{
	if (vwMean) *vwMean = g_volJVwMean;
	if (vwStd) *vwStd = g_volJVwStd;
}

// k 스윕처럼 조건을 바꿔가며 기록할 때 쓰는 한 줄 로그.
// 콘솔에 찍어두면 나중에 표로 옮기기 쉽다(UI 숫자를 손으로 받아적다 틀리는 걸 막는다).
void FORWARD::logVolumeJStats(const char* tag)
{
	printf("[J-stats]%s%s k=%d cap=%d r=%d M=%d | n=%d | 개수평균 mean=%.5f std=%.5f "
		"| 부피가중 mean=%.5f std=%.5f | range[%.4f, %.4f] p05=%.4f p95=%.4f\n",
		(tag && tag[0]) ? " " : "", (tag && tag[0]) ? tag : "",
		g_volRingK, g_volMaxMembers, g_volLeaderMinHop, g_volLeaderCount,
		g_volJStatN, g_volJMean, g_volJStd,
		g_volJVwMean, g_volJVwStd,
		g_volJMin, g_volJMax, g_volJP05, g_volJP95);
}

// ── [진단] per-Gaussian J 집계 매끄러움 측정 ────────────────────────────
//  목적: 접선-법선 분해(법선 두께 = J / det F_tan)를 쓰려면 "그 가우시안의 J"가 필요한데,
//  현재 J는 '클러스터' 단위이고 한 가우시안이 수십 개 클러스터에 겹쳐 속한다.
//  전치 CSR로 평균낸 per-Gaussian J 가 공간적으로 충분히 매끄러운지(= 이웃 간 편차가 작은지)
//  먼저 재본다. 튀면 그 J를 법선 두께에 그대로 쓰면 두께가 떨린다.
//  판정 기준: 이웃 간 |ΔJ| 가 J 자체의 변동폭(std)보다 충분히 작아야 쓸 만하다.
static void measurePerGaussianJSmoothness(const std::vector<float>& h_J,
	const std::vector<int>& h_mat, int N, const char* tag)
{
	if (N <= 0 || (int)g_hRevOffset.size() != N || g_hGraphOffset.empty()) {
		printf("[J-smooth]%s%s 측정 불가 (전치 CSR 또는 그래프 CSR 캐시 없음)\n",
			(tag && tag[0]) ? " " : "", (tag && tag[0]) ? tag : "");
		return;
	}

	// (1) per-Gaussian J = 자신을 포함하는 클러스터들의 J 평균 (겹침 평균)
	std::vector<float> Jg(N, 1.0f);
	std::vector<unsigned char> valid(N, 0);
	long long hjSum = 0; int hj0 = 0;
	for (int i = 0; i < N; ++i) {
		const int off = g_hRevOffset[i], cnt = g_hRevCount[i];
		double s = 0.0; int used = 0;
		for (int e = 0; e < cnt; ++e) {
			const int L = g_hRevIdx[off + e];
			if (L < 0 || L >= N) continue;
			if (h_mat[L] != 0) continue;              // volume 클러스터만 J가 유효
			const float j = h_J[L];
			if (!isfinite(j) || j <= 0.0f) continue;
			s += j; ++used;
		}
		hjSum += used;
		if (used > 0) { Jg[i] = (float)(s / used); valid[i] = 1; }
		else ++hj0;
	}

	// (2) J_g 자체의 분포
	double m = 0.0; int nv = 0;
	for (int i = 0; i < N; ++i) if (valid[i]) { m += Jg[i]; ++nv; }
	if (nv == 0) { printf("[J-smooth] 유효 노드 0\n"); return; }
	m /= nv;
	double v = 0.0;
	for (int i = 0; i < N; ++i) if (valid[i]) { const double d = Jg[i] - m; v += d * d; }
	const double jStd = std::sqrt(v / nv);

	// (3) ★ 매끄러움 = 그래프 이웃 간 |ΔJ_g| 분포 (원본 그래프 CSR 사용)
	std::vector<float> dj;
	dj.reserve((size_t)N * 4);
	double dSum = 0.0;
	for (int i = 0; i < N; ++i) {
		if (!valid[i]) continue;
		const int off = g_hGraphOffset[i], cnt = g_hGraphCount[i];
		for (int e = 0; e < cnt; ++e) {
			const int j = g_hGraphIdx[off + e];
			if (j < 0 || j >= N || j <= i || !valid[j]) continue;
			const float d = fabsf(Jg[i] - Jg[j]);
			dj.push_back(d); dSum += d;
		}
	}
	if (dj.empty()) { printf("[J-smooth] 이웃 쌍 0\n"); return; }
	std::sort(dj.begin(), dj.end());
	const double dMean = dSum / dj.size();
	const float dP50 = dj[dj.size() / 2];
	const float dP95 = dj[(size_t)(0.95 * (dj.size() - 1))];
	const float dMax = dj.back();
	// 상대 지표: 이웃 편차가 J 변동폭 대비 얼마나 되나. 작을수록 매끄럽다.
	const double ratio = (jStd > 1e-12) ? (dMean / jStd) : 0.0;

	printf("[J-smooth]%s%s N=%d 유효=%d (h_j=0 노드=%d, 평균 h_j=%.1f)\n",
		(tag && tag[0]) ? " " : "", (tag && tag[0]) ? tag : "",
		N, nv, hj0, (double)hjSum / std::max(1, N));
	printf("[J-smooth]   J_g: mean=%.5f std=%.5f\n", m, jStd);
	printf("[J-smooth]   이웃 |dJ|: mean=%.5f p50=%.5f p95=%.5f max=%.5f\n",
		dMean, dP50, dP95, dMax);
	printf("[J-smooth]   ★ 상대 거칠기 = mean|dJ| / std(J) = %.3f  (<0.3 매끄러움: 그대로 사용 가능 / "
		">0.6 거침: 법선 두께가 떨림 → 스무딩 필요)\n", ratio);
	// 법선 두께로 환산했을 때의 실제 영향(1차 근사: 두께비 오차 ≈ |dJ|/J)
	printf("[J-smooth]   → 이웃 간 법선두께 불일치 ≈ %.2f%% (p95 %.2f%%)\n",
		100.0 * dMean / std::max(1e-6, m), 100.0 * dP95 / std::max(1e-6, m));
}

// ── [TN] 접선-법선 분해 토글 ───────────────────────────────────────────
void FORWARD::setTangentNormalMode(bool enabled, float flatRatio)
{
	g_tnModeHost = enabled;
	g_tnFlatRatioHost = fminf(1.0f, fmaxf(0.01f, flatRatio));
	const int m = enabled ? 1 : 0;
	cudaMemcpyToSymbol(g_tnMode, &m, sizeof(int));
	cudaMemcpyToSymbol(g_tnFlatRatio, &g_tnFlatRatioHost, sizeof(float));
	const unsigned int zero2[2] = { 0u, 0u };
	cudaMemcpyToSymbol(g_tnUseCtr, zero2, sizeof(zero2));
	printf("[TN] 접선-법선 분해 = %s (flatRatio=%.3f, 법선두께는 물리 J 가 결정)\n",
		enabled ? "ON" : "OFF (기존 3D 최소제곱)", g_tnFlatRatioHost);
}

// TN 적용/폴백 카운터 조회 (UI 표시용) — 몇 %의 가우시안이 실제로 새 경로를 탔나
void FORWARD::getTangentNormalStats(unsigned int* applied, unsigned int* fallback)
{
	unsigned int h[2] = { 0u, 0u };
	cudaMemcpyFromSymbol(h, g_tnUseCtr, sizeof(h));
	if (applied) *applied = h[0];
	if (fallback) *fallback = h[1];
}

void FORWARD::resetTangentNormalStats()
{
	const unsigned int zero2[2] = { 0u, 0u };
	cudaMemcpyToSymbol(g_tnUseCtr, zero2, sizeof(zero2));
}

// ── 렌더 F 축퇴 강건화 토글 ────────────────────────────────────────────
// OFF 로 두면 기존(버그) 경로 그대로라 A/B 비교가 된다. 자세한 근거는
// g_renderFRobust 선언부 주석 참조.
static bool g_renderFRobustHost = true;

void FORWARD::setRenderFRobust(bool enabled)
{
	g_renderFRobustHost = enabled;
	const int m = enabled ? 1 : 0;
	cudaMemcpyToSymbol(g_renderFRobust, &m, sizeof(int));
	const unsigned int zero3[3] = { 0u, 0u, 0u };
	cudaMemcpyToSymbol(g_renderFCtr, zero3, sizeof(zero3));   // 조건이 바뀌었으니 통계도 리셋
	printf("[RenderF] 축퇴 강건화 = %s  (trace 정규화 det 판정 + PPt/QPt 양쪽 정규화)\n",
		enabled ? "ON" : "OFF (기존 경로: det<1e-8 이면 회전 포기)");
}

bool FORWARD::getRenderFRobust() { return g_renderFRobustHost; }

// ── [SH] 변형 회전을 겉모습에 반영 ─────────────────────────────────────
//  0 = OFF(기존) · 1 = ON(Rᵀ) · 2 = 진단용 역방향(R)
//  모양 경로와 독립이므로 OFF 로 두면 기존 결과와 비트 단위로 같다.
static int g_shRotateHost = 0;

void FORWARD::setSHRotate(int mode)
{
	g_shRotateHost = (mode < 0) ? 0 : ((mode > 2) ? 2 : mode);
	cudaMemcpyToSymbol(g_shRotate, &g_shRotateHost, sizeof(int));
	const unsigned int zero2[2] = { 0u, 0u };
	cudaMemcpyToSymbol(g_shRotCtr, zero2, sizeof(zero2));
	printf("[SH] 변형 회전 반영 = %s\n",
		g_shRotateHost == 0 ? "OFF (SH 는 월드 고정 — 기존)" :
		(g_shRotateHost == 1 ? "ON (dir <- R^T dir)" : "진단: 역방향 (dir <- R dir)"));
}

int FORWARD::getSHRotate() { return g_shRotateHost; }

// [0]=회전 적용, [1]=폴백(축퇴/반전이라 항등 유지)
void FORWARD::getSHRotStats(unsigned int* applied, unsigned int* fallback)
{
	unsigned int h[2] = { 0u, 0u };
	cudaMemcpyFromSymbol(h, g_shRotCtr, sizeof(h));
	if (applied)  *applied = h[0];
	if (fallback) *fallback = h[1];
}

void FORWARD::resetSHRotStats()
{
	const unsigned int zero2[2] = { 0u, 0u };
	cudaMemcpyToSymbol(g_shRotCtr, zero2, sizeof(zero2));
}

// [0]=변형 적용, [1]=det 실패(회전 멈춤), [2]=이웃<3(회전 멈춤)
void FORWARD::getRenderFStats(unsigned int* applied, unsigned int* detFail, unsigned int* fewNbr)
{
	unsigned int h[3] = { 0u, 0u, 0u };
	cudaMemcpyFromSymbol(h, g_renderFCtr, sizeof(h));
	if (applied) *applied = h[0];
	if (detFail) *detFail = h[1];
	if (fewNbr)  *fewNbr = h[2];
}

void FORWARD::resetRenderFStats()
{
	const unsigned int zero3[3] = { 0u, 0u, 0u };
	cudaMemcpyToSymbol(g_renderFCtr, zero3, sizeof(zero3));
}

// UI 버튼용: 다음 프레임 J 계산 직후에 1회 측정하도록 예약한다.
// (h_J/h_mat 가 그 자리에서만 살아 있으므로 즉시 실행이 아니라 예약 방식)
void FORWARD::requestJSmoothnessLog(const char* tag)
{
	g_jSmoothPending = true;
	snprintf(g_jSmoothTag, sizeof(g_jSmoothTag), "%s", (tag && tag[0]) ? tag : "manual");
	printf("[J-smooth] 다음 프레임에 측정 예약 (J Stats가 켜져 있어야 함)\n");
}

// ── Squash Test ─────────────────────────────────────────
static void freeSquashBuffers()
{
	if (d_squashTopIdx) { cudaFree(d_squashTopIdx); d_squashTopIdx = nullptr; }
	if (d_squashBotIdx) { cudaFree(d_squashBotIdx); d_squashBotIdx = nullptr; }
	if (d_squashTopRest) { cudaFree(d_squashTopRest); d_squashTopRest = nullptr; }
	if (d_squashBotRest) { cudaFree(d_squashBotRest); d_squashBotRest = nullptr; }
	g_squashTopCount = 0;
	g_squashBotCount = 0;
}

void FORWARD::squashStart(int axis, float slabPct, float rampPerSec, float maxDispPct)
{
	if (!d_pos_rest || cm_num_elements <= 0) {
		printf("[Squash] 그래프가 로드되지 않았다.\n");
		return;
	}
	if (axis < 0 || axis > 2) axis = 1;
	slabPct = fmaxf(0.01f, fminf(0.5f, slabPct));
	maxDispPct = fmaxf(0.01f, fminf(1.0f, maxDispPct));

	g_squashAxis = axis;
	g_squashSlabPct = slabPct;
	g_squashRampPerSec = rampPerSec;

	const int N = cm_num_elements;
	std::vector<float3> h_rest(N);
	cudaMemcpy(h_rest.data(), d_pos_rest, sizeof(float3) * N, cudaMemcpyDeviceToHost);

	// 축 방향 min/max 찾기
	float lo = FLT_MAX, hi = -FLT_MAX;
	for (int i = 0; i < N; ++i) {
		const float v = (axis == 0) ? h_rest[i].x : (axis == 1) ? h_rest[i].y : h_rest[i].z;
		lo = fminf(lo, v); hi = fmaxf(hi, v);
	}
	const float extent = hi - lo;
	g_squashSceneExtent = extent;
	g_squashPress = g_squashPressNext;
	g_squashLo = lo;
	g_squashHi = hi;
	g_squashMaxDisp = maxDispPct * extent;
	g_squashCurDisp = 0.0f;

	const float topThr = hi - slabPct * extent; // 이상이면 상단
	const float botThr = lo + slabPct * extent; // 이하면 하단

	std::vector<int> topIdx; topIdx.reserve(N / 8);
	std::vector<int> botIdx; botIdx.reserve(N / 8);
	std::vector<float3> topRest; topRest.reserve(N / 8);
	std::vector<float3> botRest; botRest.reserve(N / 8);
	for (int i = 0; i < N; ++i) {
		const float v = (axis == 0) ? h_rest[i].x : (axis == 1) ? h_rest[i].y : h_rest[i].z;
		if (v >= topThr) { topIdx.push_back(i); topRest.push_back(h_rest[i]); }
		else if (v <= botThr) { botIdx.push_back(i); botRest.push_back(h_rest[i]); }
	}

	freeSquashBuffers();
	g_squashTopCount = (int)topIdx.size();
	g_squashBotCount = (int)botIdx.size();

	if (g_squashTopCount == 0 || g_squashBotCount == 0) {
		printf("[Squash] 슬랩이 비었다. axis=%d slabPct=%.2f 로 top=%d bot=%d\n",
			axis, slabPct, g_squashTopCount, g_squashBotCount);
		return;
	}

	cudaMalloc(&d_squashTopIdx, sizeof(int) * g_squashTopCount);
	cudaMalloc(&d_squashBotIdx, sizeof(int) * g_squashBotCount);
	cudaMalloc(&d_squashTopRest, sizeof(float3) * g_squashTopCount);
	cudaMalloc(&d_squashBotRest, sizeof(float3) * g_squashBotCount);
	cudaMemcpy(d_squashTopIdx, topIdx.data(), sizeof(int) * g_squashTopCount, cudaMemcpyHostToDevice);
	cudaMemcpy(d_squashBotIdx, botIdx.data(), sizeof(int) * g_squashBotCount, cudaMemcpyHostToDevice);
	cudaMemcpy(d_squashTopRest, topRest.data(), sizeof(float3) * g_squashTopCount, cudaMemcpyHostToDevice);
	cudaMemcpy(d_squashBotRest, botRest.data(), sizeof(float3) * g_squashBotCount, cudaMemcpyHostToDevice);

	g_squashActive = true;
	g_femMetaFresh = true;   // 이 실행의 첫 체크포인트에서 metadata 를 새로 쓴다
	const char* axName = (axis == 0) ? "X" : (axis == 1) ? "Y" : "Z";
	// 자동 로깅 상태 초기화 — 매 START마다 체크포인트를 처음부터 다시 건다.
	g_squashCkptIdx = 0;
	g_squashDwell = 0;
	g_squashLogPending = false;

	printf("[Squash] START axis=%s extent=%.3f slab=%.1f%% (top=%d, bot=%d) maxDisp=%.3f rampPerSec=%.3f\n",
		axName, extent, slabPct * 100.0f, g_squashTopCount, g_squashBotCount,
		g_squashMaxDisp, rampPerSec);
	if (g_squashPress) {
		printf("[Squash] PRESS mode: slabs are NOT pinned. %s plate moves (rest lo %.4f, hi %.4f), contact = floor-style projection (frictionless)\n",
			g_squashPressFromLo ? "lo" : "hi",
			lo, hi);
	}
	if (g_squashAutoLog) {
		printf("[Squash] 자동 로깅 ON: 변위 25/50/75/100%%에서 %d프레임 정착 후 J 기록 (k 스윕 공정 비교용)\n",
			g_squashDwellFrames);
		if (!g_volCollectJStats)
			printf("[Squash] ★ 경고: J Stats가 꺼져 있다. 켜지 않으면 체크포인트가 기록되지 않는다.\n");
	}
}

void FORWARD::squashStop()
{
	if (!g_squashActive) return;
	g_squashActive = false;
	// 핀을 풀어놓기만 하고 위치는 그대로 둔다 (물체가 자연스럽게 회복하는 걸 관찰할 수 있게).
	if (d_invMass && cm_num_elements > 0) {
		const int N = cm_num_elements;
		xpbdResetInvMassKernel << <(N + 255) / 256, 256 >> > (N, d_invMass, 1.0f);
		cudaDeviceSynchronize();
	}
	printf("[Squash] STOP (curDisp=%.4f)\n", g_squashCurDisp);
}

void FORWARD::squashReset()
{
	// 완전 리셋: 씬을 rest로 되돌리고 pin 해제.
	if (!d_pos_rest || cm_num_elements <= 0) return;
	const int N = cm_num_elements;
	cudaMemcpy(d_pos_curr, d_pos_rest, sizeof(float3) * N, cudaMemcpyDeviceToDevice);
	cudaMemset(d_vel, 0, sizeof(float3) * N);
	xpbdResetInvMassKernel << <(N + 255) / 256, 256 >> > (N, d_invMass, 1.0f);
	if (d_lambda_vol) cudaMemset(d_lambda_vol, 0, sizeof(float) * N);
	cudaDeviceSynchronize();
	g_squashActive = false;
	g_squashCurDisp = 0.0f;
	g_squashCkptIdx = 0;
	g_squashDwell = 0;
	g_squashLogPending = false;
	printf("[Squash] RESET: 씬을 rest 상태로 복원\n");
}

// ── Ground / drop demo 호스트 API ──────────────────────────────────────────
void FORWARD::setGroundParams(bool enabled, const float up[3], float height, float friction, float restitution,
	float contactRadius, float contactSlop, float gravity, bool paused)
{
	float nx = up ? up[0] : 0.0f, ny = up ? up[1] : 0.0f, nz = up ? up[2] : 1.0f;
	const float len = sqrtf(nx * nx + ny * ny + nz * nz);
	if (!(len > 1e-8f) || !isfinite(len)) { nx = 0.0f; ny = 0.0f; nz = 1.0f; }
	else { nx /= len; ny /= len; nz /= len; }
	g_groundN[0] = nx; g_groundN[1] = ny; g_groundN[2] = nz;
	g_groundEnabled = enabled;
	g_groundHeight = isfinite(height) ? height : 0.0f;
	g_groundFriction = fmaxf(0.0f, friction);
	g_groundRestitution = fminf(fmaxf(restitution, 0.0f), 1.0f);
	g_groundRadius = fmaxf(0.0f, contactRadius);
	g_groundSlop = fmaxf(0.0f, contactSlop);
	g_groundGravity = fmaxf(0.0f, gravity);
	g_groundPaused = paused;
}

void FORWARD::setGroundRender(bool visible, const float origin[3], float checkerSize, float fadeRadius,
	const float* viewmatrix, const float campos[3], float tanFovX, float tanFovY)
{
	g_groundVisible = visible && origin && viewmatrix && campos;
	if (!g_groundVisible) return;
	for (int i = 0; i < 3; ++i) {
		g_groundOrigin[i] = origin[i];
		g_groundCamPos[i] = campos[i];
	}
	for (int i = 0; i < 16; ++i) g_groundViewM[i] = viewmatrix[i];
	g_groundChecker = fmaxf(checkerSize, 1e-8f);
	g_groundFadeRadius = fmaxf(fadeRadius, 1e-8f);
	g_groundTanFovX = tanFovX;
	g_groundTanFovY = tanFovY;
}

void FORWARD::setObjectModeParams(float shapeStiffness, bool bodyContact, int minComponent)
{
	float s = fminf(fmaxf(shapeStiffness, 0.0f), 1.0f);
	if (!isfinite(s)) s = 0.0f;
	if ((g_objShapeStiffness > 0.0f) != (s > 0.0f) || g_objBodyContact != bodyContact) g_objPrevValid = false;
	g_objShapeStiffness = s;
	g_objBodyContact = bodyContact;
	const int mc = (minComponent < 1) ? 1 : minComponent;
	if (mc != g_objMinComponent) {
		g_objMinComponent = mc;     // 다음 프레임에 연결 성분을 다시 나눈다
		g_objPrevValid = false;
	}
}

void FORWARD::setKinematicColliders(int count, const int* types, const float* poses, const float* dims, float margin,
	float friction)
{
	if (count < 0) count = 0;
	if (count > MAX_KIN_COLLIDERS) {
		printf("[Collider] %d colliders requested, using the first %d\n", count, MAX_KIN_COLLIDERS);
		count = MAX_KIN_COLLIDERS;
	}
	const bool samePrev = (count == g_kinColCount);   // 같은 목록이면 직전 자세 = 지난번 값 (마찰이 표면 움직임을 뺀다)
	for (int c = 0; c < count; ++c) {
		KinCollider& K = g_kinCol[c];
		const bool keep = samePrev && K.type == types[c];
		for (int a = 0; a < 9; ++a) K.Rp[a] = keep ? K.R[a] : poses[12 * c + a];
		for (int a = 0; a < 3; ++a) K.tp[a] = keep ? K.t[a] : poses[12 * c + 9 + a];
		K.type = types[c];
		for (int a = 0; a < 9; ++a) K.R[a] = poses[12 * c + a];
		for (int a = 0; a < 3; ++a) { K.t[a] = poses[12 * c + 9 + a]; K.h[a] = fmaxf(dims[3 * c + a], 0.0f); }
	}
	g_kinColCount = count;
	g_kinColMargin = (isfinite(margin) && margin > 0.0f) ? margin : 0.0f;
	g_kinColFriction = (isfinite(friction) && friction > 0.0f) ? friction : 0.0f;
	g_kinColDirty = true;
}

int FORWARD::getKinematicColliderStats(int* hits, float* push, float* centroid, int maxCount)
{
	const int n = (maxCount < g_kinColCount) ? maxCount : g_kinColCount;
	if (n <= 0 || !d_kinColHits) return 0;
	if (hits) cudaMemcpy(hits, d_kinColHits, sizeof(int) * n, cudaMemcpyDeviceToHost);
	if (push || centroid) {
		std::vector<float> acc(KINCOL_ACC * (size_t)n);
		cudaMemcpy(acc.data(), d_kinColPush, sizeof(float) * KINCOL_ACC * n, cudaMemcpyDeviceToHost);
		for (int c = 0; c < n; ++c) {
			const float* a = &acc[KINCOL_ACC * c];
			for (int k = 0; k < 3; ++k) {
				if (push) push[3 * c + k] = a[k];
				if (centroid) centroid[3 * c + k] = (a[6] > 0.0f) ? a[3 + k] / a[6] : g_kinCol[c].t[k];   // 접촉 없으면 충돌체 중심
			}
		}
	}
	return n;
}

void FORWARD::setObjectShapeGPU(bool enabled) { g_objShapeGPU = enabled; }
bool FORWARD::getObjectShapeGPU() { return g_objShapeGPU; }

void FORWARD::getObjectComponentStats(int* objects, int* largest, int* smallComponents, int* smallParticles)
{
	if (objects) *objects = g_objNumComponents;
	if (largest) { largest[0] = g_objLargest[0]; largest[1] = g_objLargest[1]; largest[2] = g_objLargest[2]; }
	if (smallComponents) *smallComponents = g_objNumSmall;
	if (smallParticles) *smallParticles = g_objSmallParticles;
}

void FORWARD::setSelfCollisionParams(bool enabled, float radiusScale, float excludeScale, bool withinBody)
{
	g_selfColEnabled = enabled;
	g_selfColWithinBody = withinBody;
	g_selfColRadiusScale = fminf(fmaxf(isfinite(radiusScale) ? radiusScale : 1.5f, 0.1f), 10.0f);
	// d_ex ≤ d_c 면 rest에서 이미 겹친 쌍이 목록에 들어와 가만히 있어도 터진다 → 하한 1.05
	g_selfColExcludeScale = fminf(fmaxf(isfinite(excludeScale) ? excludeScale : 2.0f, 1.05f), 20.0f);
	if (!enabled) { g_selfColContacts = 0; g_selfColOverflow = 0; }
}

void FORWARD::getSelfCollisionStats(float* spacing, float* contactDist, int* contacts, int* overflow, int* active)
{
	if (active) *active = g_selfColActive;
	if (spacing) *spacing = g_selfColSpacing;
	if (contactDist) *contactDist = g_selfColRadiusScale * g_selfColSpacing;
	if (contacts) *contacts = g_selfColContacts;
	if (overflow) *overflow = g_selfColOverflow;
}

bool FORWARD::groundLaunch(const float linVel[3], const float angVel[3], const float tiltAxisAngle[3])
{
	if (!d_pos_rest || !d_pos_curr || !d_vel || !d_invMass || cm_num_elements <= 0) {
		printf("[Ground] 그래프가 로드되지 않았다.\n");
		return false;
	}
	const int N = cm_num_elements;
	std::vector<float3> h_rest(N);
	cudaMemcpy(h_rest.data(), d_pos_rest, sizeof(float3) * N, cudaMemcpyDeviceToHost);

	double cx = 0.0, cy = 0.0, cz = 0.0;
	for (int i = 0; i < N; ++i) { cx += h_rest[i].x; cy += h_rest[i].y; cz += h_rest[i].z; }
	const float comX = float(cx / N), comY = float(cy / N), comZ = float(cz / N);

	// Rodrigues: R = cosθ·I + sinθ·[k]× + (1−cosθ)·kkᵀ  (행우선 R[r*3+c])
	// 거리·형상·부피 제약은 모두 회전 불변이라, rest를 돌린 자세에서 출발해도 '변형'으로 보지 않는다.
	const float ax = tiltAxisAngle ? tiltAxisAngle[0] : 0.0f;
	const float ay = tiltAxisAngle ? tiltAxisAngle[1] : 0.0f;
	const float az = tiltAxisAngle ? tiltAxisAngle[2] : 0.0f;
	const float th = sqrtf(ax * ax + ay * ay + az * az);
	float R[9] = { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f };
	if (th > 1e-8f) {
		const float kx = ax / th, ky = ay / th, kz = az / th;
		const float s = sinf(th), c = cosf(th), t = 1.0f - c;
		R[0] = c + t * kx * kx;      R[1] = t * kx * ky - s * kz; R[2] = t * kx * kz + s * ky;
		R[3] = t * ky * kx + s * kz; R[4] = c + t * ky * ky;      R[5] = t * ky * kz - s * kx;
		R[6] = t * kz * kx - s * ky; R[7] = t * kz * ky + s * kx; R[8] = c + t * kz * kz;
	}
	const float vx = linVel ? linVel[0] : 0.0f, vy = linVel ? linVel[1] : 0.0f, vz = linVel ? linVel[2] : 0.0f;
	const float wx = angVel ? angVel[0] : 0.0f, wy = angVel ? angVel[1] : 0.0f, wz = angVel ? angVel[2] : 0.0f;

	std::vector<float3> h_pos(N), h_vel(N);
	for (int i = 0; i < N; ++i) {
		const float px = h_rest[i].x - comX, py = h_rest[i].y - comY, pz = h_rest[i].z - comZ;
		const float rx = R[0] * px + R[1] * py + R[2] * pz;
		const float ry = R[3] * px + R[4] * py + R[5] * pz;
		const float rz = R[6] * px + R[7] * py + R[8] * pz;
		h_pos[i] = make_float3(comX + rx, comY + ry, comZ + rz);
		// 강체 속도장 v = v_lin + ω × r
		h_vel[i] = make_float3(vx + (wy * rz - wz * ry), vy + (wz * rx - wx * rz), vz + (wx * ry - wy * rx));
	}
	cudaMemcpy(d_pos_curr, h_pos.data(), sizeof(float3) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vel, h_vel.data(), sizeof(float3) * N, cudaMemcpyHostToDevice);
	xpbdResetInvMassKernel << <(N + 255) / 256, 256 >> > (N, d_invMass, 1.0f);
	if (d_lambda_vol) cudaMemset(d_lambda_vol, 0, sizeof(float) * N);
	cudaDeviceSynchronize();

	// squash 하중이 걸려 있으면 슬랩이 물체를 붙잡으므로 해제한다.
	g_squashActive = false;
	g_squashCurDisp = 0.0f;
	g_squashCkptIdx = 0;
	g_squashDwell = 0;
	g_squashLogPending = false;
	g_objPrevValid = false;   // 순간이동한 상태라 직전 강체 속도로 충돌 전 속도를 추정하면 안 된다
	g_impactLogLeft = 150;    // 충돌 과도응답 로그 (2.5초 @ 60fps, 3프레임마다)
	g_impactLogFrame = 0;
	printf("[Ground] launch: N=%d |v|=%.4g |w|=%.4g rad/s tilt=%.1f deg\n", N,
		sqrtf(vx * vx + vy * vy + vz * vz), sqrtf(wx * wx + wy * wy + wz * wz), th * 57.29578f);
	return true;
}

// ── Isaac Sim 연동: 물리 그래프 내보내기 ─────────────────────────────────────
// GPU 에 올라간 CSR 과 rest 위치를 비트 그대로 쓴다. 순서는 뷰어 내부 순서(loadPly 의 Morton 정렬 → 크롭)라서
// tools/import_graph.py 가 rest 위치를 비트 단위로 맞춰 USD(원본 PLY 크롭) 순서로 바꾼다.
// 각도 짝(pairNextIdx·restCos)과 부피 클러스터는 이 CSR 과 위치로 다시 만들어지므로 쓰지 않는다.
// 형식 (little endian): "APGGRPH1" | int32 N | int32 M | float32 pos_rest[N*3] | int32 offset[N] | int32 count[N]
//                       | int32 idx[M] | float32 dist[M] | float32 stiff[M]
bool FORWARD::exportPhysicsGraph(const char* path)
{
	const int N = cm_num_elements;
	const int M = cm_num_neighbors;
	if (!path || !d_pos_rest || !d_offset || !d_nbrCount || !d_nbrIdx || !d_nbrDist || !d_nbrStiff || N <= 0 || M <= 0) {
		printf("[Export] 물리 그래프가 아직 GPU 에 없다 (XPBD 모드에서 한 프레임 이상 돈 뒤에 누를 것)\n");
		return false;
	}
	std::vector<float3> pos(N);
	std::vector<int> off(N), cnt(N), idx(M);
	std::vector<float> dist(M), st(M);
	cudaMemcpy(pos.data(), d_pos_rest, sizeof(float3) * N, cudaMemcpyDeviceToHost);
	cudaMemcpy(off.data(), d_offset, sizeof(int) * N, cudaMemcpyDeviceToHost);
	cudaMemcpy(cnt.data(), d_nbrCount, sizeof(int) * N, cudaMemcpyDeviceToHost);
	cudaMemcpy(idx.data(), d_nbrIdx, sizeof(int) * M, cudaMemcpyDeviceToHost);
	cudaMemcpy(dist.data(), d_nbrDist, sizeof(float) * M, cudaMemcpyDeviceToHost);
	cudaMemcpy(st.data(), d_nbrStiff, sizeof(float) * M, cudaMemcpyDeviceToHost);

	std::ofstream f(path, std::ios::binary);
	if (!f) {
		printf("[Export] 파일을 열 수 없다: %s\n", path);
		return false;
	}
	f.write("APGGRPH1", 8);
	f.write(reinterpret_cast<const char*>(&N), sizeof(int));
	f.write(reinterpret_cast<const char*>(&M), sizeof(int));
	f.write(reinterpret_cast<const char*>(pos.data()), sizeof(float3) * N);
	f.write(reinterpret_cast<const char*>(off.data()), sizeof(int) * N);
	f.write(reinterpret_cast<const char*>(cnt.data()), sizeof(int) * N);
	f.write(reinterpret_cast<const char*>(idx.data()), sizeof(int) * M);
	f.write(reinterpret_cast<const char*>(dist.data()), sizeof(float) * M);
	f.write(reinterpret_cast<const char*>(st.data()), sizeof(float) * M);
	f.close();
	if (!f) {
		printf("[Export] 쓰기 실패: %s\n", path);
		return false;
	}
	int isolated = 0, maxDeg = 0;
	for (int i = 0; i < N; ++i) {
		if (cnt[i] <= 0) ++isolated;
		maxDeg = std::max(maxDeg, cnt[i]);
	}
	printf("[Export] %s | nodes %d | CSR entries %d (avg degree %.2f, max %d, isolated %d)\n",
		path, N, M, (double)M / N, maxDeg, isolated);
	return true;
}

bool FORWARD::squashIsActive() { return g_squashActive; }
// 완료된 체크포인트 수 (0~4). 4면 이번 실행의 덤프가 모두 끝난 것이므로
// 자동 스윕이 다음 설정으로 넘어갈 수 있다.
int FORWARD::squashCheckpointsDone() { return g_squashCkptIdx; }
void FORWARD::squashSetSlip(bool slip) { g_squashSlip = slip; }
void FORWARD::squashSetPress(bool press) { g_squashPressNext = press; }   // 다음 Start 부터 적용
void FORWARD::squashSetPressFromLo(bool fromLo) { g_squashPressFromLo = fromLo; }
// 25/50/75/100% 체크포인트 정착(멈춤 + J 기록) on/off. 뷰어의 FEM 비교 실험은 켜 둔다(기본). 데모는 끄고 쭉 누른다.
void FORWARD::squashSetAutoLog(bool enabled) { g_squashAutoLog = enabled; }
// 현재 평판 위치 (축 좌표). press 가 아니거나 멈춘 뒤에는 rest 경계 그대로.
void FORWARD::squashGetPlates(float* lo, float* hi)
{
	const bool moving = g_squashActive && g_squashPress;
	if (lo) *lo = g_squashLo + ((moving && g_squashPressFromLo) ? g_squashCurDisp : 0.0f);
	if (hi) *hi = g_squashHi - ((moving && !g_squashPressFromLo) ? g_squashCurDisp : 0.0f);
}
bool FORWARD::squashGetSlip() { return g_squashSlip; }
void FORWARD::squashGetProgress(float* curDisp, float* maxDisp, int* axis, int* topCnt, int* botCnt)
{
	if (curDisp) *curDisp = g_squashCurDisp;
	if (maxDisp) *maxDisp = g_squashMaxDisp;
	if (axis) *axis = g_squashAxis;
	if (topCnt) *topCnt = g_squashTopCount;
	if (botCnt) *botCnt = g_squashBotCount;
}

void FORWARD::setFEMBenchmarkDumpEnabled(bool enabled, const std::string& directory, const std::string& runLabel)
{
	g_femBenchmarkDumpEnabled = enabled;
	if (!directory.empty()) g_femBenchmarkDumpDir = directory;
	if (!runLabel.empty()) g_femBenchmarkRunLabel = sanitizeFEMLabel(runLabel);
	if (g_femBenchmarkDumpEnabled) {
		ensureFEMDirectory(g_femBenchmarkDumpDir);
		printf("[FEM benchmark] dumps %s | dir=%s | label=%s\n",
			enabled ? "ON" : "OFF", g_femBenchmarkDumpDir.c_str(), g_femBenchmarkRunLabel.c_str());
	}
}

bool FORWARD::getFEMBenchmarkDumpEnabled()
{
	return g_femBenchmarkDumpEnabled;
}

static void dumpFEMBenchmarkCheckpoint()
{
	if (!g_femBenchmarkDumpEnabled || !g_squashActive || cm_num_elements <= 0) return;
	if (!d_pos_curr || g_squashSceneExtent <= 1.0e-8f) return;

	ensureFEMDirectory(g_femBenchmarkDumpDir);
	const int N = cm_num_elements;
	std::vector<float3> h_pos(static_cast<size_t>(N));
	const cudaError_t copyErr = cudaMemcpy(
		h_pos.data(), d_pos_curr, sizeof(float3) * static_cast<size_t>(N), cudaMemcpyDeviceToHost);
	if (copyErr != cudaSuccess) {
		printf("[FEM benchmark] position copy failed: %s\n", cudaGetErrorString(copyErr));
		return;
	}

	const float normalizedStrain = g_squashCurDisp / g_squashSceneExtent;
	char fileName[256];
	snprintf(fileName, sizeof(fileName), "%s_%s_strain_%.6f.csv",
		g_femBenchmarkRunLabel.c_str(),
		(g_squashAxis == 0) ? "x" : (g_squashAxis == 1) ? "y" : "z",
		normalizedStrain);
	const std::string path = g_femBenchmarkDumpDir + "/" + fileName;
	std::ofstream out(path, std::ios::out | std::ios::trunc);
	if (!out.is_open()) {
		printf("[FEM benchmark] cannot open dump: %s\n", path.c_str());
		return;
	}
	out << "index,x,y,z\n" << std::setprecision(9);
	for (int i = 0; i < N; ++i) {
		out << i << ',' << h_pos[i].x << ',' << h_pos[i].y << ',' << h_pos[i].z << '\n';
	}
	out.close();

	// Keep the physical/runtime settings next to the coordinate dump. This is
	// the label needed later when several c_vol/compliance runs are calibrated.
	const std::string metaPath = g_femBenchmarkDumpDir + "/" + g_femBenchmarkRunLabel + "_metadata.csv";
	const bool freshRun = g_femMetaFresh;
	g_femMetaFresh = false;
	const bool newMeta = freshRun || !std::ifstream(metaPath).good();
	std::ofstream meta(metaPath,
		std::ios::out | (freshRun ? std::ios::trunc : std::ios::app));
	if (meta.is_open()) {
		if (newMeta) {
			meta << "strain,axis,disp,scene_extent,solver_iters,dt,under_relax,vel_damping,"
				"distance_compliance,shape_compliance,volume_compliance,physical_material,E,nu,c_vol,"
				"vol_ring_k,vol_max_members,vol_leader_hop,vol_leaders,platen_slip,"
				"dist_on,dist_gs,vol_on,vol_gs,vol_gather,vol_warp_mode,shape_matching,angle,gnh,"
				"obj_shape,ramp_per_sec,press\n";
		}
		meta << std::setprecision(9)
			<< normalizedStrain << ',' << g_squashAxis << ',' << g_squashCurDisp << ','
			<< g_squashSceneExtent << ',' << g_xpbdSolverIters << ',' << g_xpbdDt << ','
			<< g_xpbdUnderRelax << ',' << g_xpbdVelDamping << ',' << g_xpbdDistanceCompliance << ','
			<< g_xpbdShapeCompliance << ',' << g_volCompliance << ','
			<< (g_volUsePhysicalAlpha ? 1 : 0) << ',' << g_volMatE << ',' << g_volMatNu << ','
			<< g_volCvol << ','
			<< g_volRingK << ',' << g_volMaxMembers << ',' << g_volLeaderMinHop << ','
			<< g_volLeaderCount << ',' << (g_squashSlip ? 1 : 0) << ','
			// 어떤 제약이 켜져 있었는지 — 이게 없으면 표의 솔버 구성을 사후에 확정할 수 없다
			<< (g_useXPBDDistanceConstraint ? 1 : 0) << ',' << (g_useDistanceGS ? 1 : 0) << ','
			<< (g_useVolumeConstraint ? 1 : 0) << ',' << (g_useVolumeGS ? 1 : 0) << ','
			<< g_volGatherMode << ',' << g_volWarpMode << ','
			<< (g_useXPBDShapeMatching ? 1 : 0) << ',' << (g_useXPBDAngleConstraint ? 1 : 0) << ','
			<< (g_useGaussianNH ? 1 : 0) << ',' << g_objShapeStiffness << ','
			<< g_squashRampPerSec << ',' << (g_squashPress ? 1 : 0) << '\n';
	}
	printf("[FEM benchmark] wrote strain=%.6f nodes=%d -> %s\n",
		normalizedStrain, N, path.c_str());
}

void FORWARD::setRegionBalloonEnabled(bool enabled)
{
	g_useRegionBalloon = enabled;
	if (!enabled) {
		g_regionBalloonCount = 0;
		g_regionBalloonDetRest = 0.0f;
		g_regionBalloonRestScale = 0.0f;
		g_regionBalloonAnchor = -1;
	}
	printf("[RegionBalloon] constraint %s\n", enabled ? "ON" : "OFF");
}

bool FORWARD::getRegionBalloonEnabled()
{
	return g_useRegionBalloon;
}

void FORWARD::setRegionBalloonParams(float compliance, float strength, float maxStep, int hops)
{
	g_regionBalloonCompliance = fmaxf(0.0f, compliance);
	g_regionBalloonStrength = fmaxf(0.0f, strength);
	g_regionBalloonMaxStep = fminf(1.0f, fmaxf(0.001f, maxStep));
	g_regionBalloonHops = (hops < 1) ? 1 : ((hops > 16) ? 16 : hops);
}

void FORWARD::getRegionBalloonParams(float* compliance, float* strength, float* maxStep, int* hops)
{
	if (compliance) *compliance = g_regionBalloonCompliance;
	if (strength) *strength = g_regionBalloonStrength;
	if (maxStep) *maxStep = g_regionBalloonMaxStep;
	if (hops) *hops = g_regionBalloonHops;
}

int FORWARD::getRegionBalloonCount()
{
	return g_regionBalloonCount;
}

void FORWARD::setChainmailParams(
	int propIters,
	int relaxIters,
	float propStrength,
	float stiffness,
	float damping)
{
	g_cmPropIters = (propIters < 0) ? 0 : propIters;
	g_cmRelaxIters = (relaxIters < 0) ? 0 : relaxIters;
	g_cmPropStrength = fmaxf(0.0f, propStrength);
	g_cmStiffness = fmaxf(0.0f, stiffness);
	g_cmDamping = fmaxf(0.0f, damping);
}

void FORWARD::getChainmailParams(
	int* propIters,
	int* relaxIters,
	float* propStrength,
	float* stiffness,
	float* damping)
{
	if (propIters) *propIters = g_cmPropIters;
	if (relaxIters) *relaxIters = g_cmRelaxIters;
	if (propStrength) *propStrength = g_cmPropStrength;
	if (stiffness) *stiffness = g_cmStiffness;
	if (damping) *damping = g_cmDamping;
}

void FORWARD::setChainmailMaterialParams(
	float constraintGlobalScale,
	float airScale,
	float skinScale,
	float boneScale,
	bool useEdgeStiffness,
	float edgeStiffnessInfluence)
{
	g_cmConstraintGlobalScale = fmaxf(1e-4f, constraintGlobalScale);
	g_cmConstraintAirScale = fmaxf(1e-4f, airScale);
	g_cmConstraintSkinScale = fmaxf(1e-4f, skinScale);
	g_cmConstraintBoneScale = fmaxf(1e-4f, boneScale);
	g_cmUseEdgeStiffness = useEdgeStiffness;
	g_cmEdgeStiffnessInfluence = fmaxf(0.0f, edgeStiffnessInfluence);
}

void FORWARD::getChainmailMaterialParams(
	float* constraintGlobalScale,
	float* airScale,
	float* skinScale,
	float* boneScale,
	bool* useEdgeStiffness,
	float* edgeStiffnessInfluence)
{
	if (constraintGlobalScale) *constraintGlobalScale = g_cmConstraintGlobalScale;
	if (airScale) *airScale = g_cmConstraintAirScale;
	if (skinScale) *skinScale = g_cmConstraintSkinScale;
	if (boneScale) *boneScale = g_cmConstraintBoneScale;
	if (useEdgeStiffness) *useEdgeStiffness = g_cmUseEdgeStiffness;
	if (edgeStiffnessInfluence) *edgeStiffnessInfluence = g_cmEdgeStiffnessInfluence;
}

void FORWARD::setChainmailDynamicsParams(
	float inertiaGain,
	float velocityRetention,
	float velocityClamp)
{
	g_cmInertiaGain = fmaxf(0.0f, inertiaGain);
	g_cmVelocityRetention = fminf(fmaxf(0.0f, velocityRetention), 0.9999f);
	g_cmVelocityClamp = fmaxf(1e-5f, velocityClamp);
}

void FORWARD::getChainmailDynamicsParams(
	float* inertiaGain,
	float* velocityRetention,
	float* velocityClamp)
{
	if (inertiaGain) *inertiaGain = g_cmInertiaGain;
	if (velocityRetention) *velocityRetention = g_cmVelocityRetention;
	if (velocityClamp) *velocityClamp = g_cmVelocityClamp;
}

void FORWARD::setXPBDParams(
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
	float angleBlend)
{
	g_xpbdSolverIters = (solverIters < 1) ? 1 : solverIters;
	g_xpbdDt = fmaxf(1e-5f, dt);
	g_xpbdUnderRelax = fmaxf(0.0f, underRelax);
	g_xpbdVelDamping = fmaxf(0.0f, velDamping);
	g_xpbdStiffnessScale = fmaxf(0.0f, stiffnessScale);
	g_xpbdInvMassScale = fmaxf(0.0f, invMassScale);
	g_xpbdDistanceCompliance = fmaxf(0.0f, distanceCompliance);
	g_xpbdShapeCompliance = fmaxf(0.0f, shapeCompliance);
	// Keep blend in a stable range [0, 1]. Higher values easily overshoot with multiple constraints.
	g_xpbdShapeBlend = fminf(fmaxf(0.0f, shapeBlend), 1.0f);
	g_xpbdAngleCompliance = fmaxf(0.0f, angleCompliance);
	g_xpbdAngleBlend = fminf(fmaxf(0.0f, angleBlend), 1.0f);
}

void FORWARD::getXPBDParams(
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
	float* angleBlend)
{
	if (solverIters) *solverIters = g_xpbdSolverIters;
	if (dt) *dt = g_xpbdDt;
	if (underRelax) *underRelax = g_xpbdUnderRelax;
	if (velDamping) *velDamping = g_xpbdVelDamping;
	if (stiffnessScale) *stiffnessScale = g_xpbdStiffnessScale;
	if (invMassScale) *invMassScale = g_xpbdInvMassScale;
	if (distanceCompliance) *distanceCompliance = g_xpbdDistanceCompliance;
	if (shapeCompliance) *shapeCompliance = g_xpbdShapeCompliance;
	if (shapeBlend) *shapeBlend = g_xpbdShapeBlend;
	if (angleCompliance) *angleCompliance = g_xpbdAngleCompliance;
	if (angleBlend) *angleBlend = g_xpbdAngleBlend;
}

__device__ __forceinline__ float3 f3_add(const float3& a, const float3& b) {
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ __forceinline__ float3 f3_sub(const float3& a, const float3& b) {
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ __forceinline__ float3 f3_mul(const float3& a, float s) {
	return make_float3(a.x * s, a.y * s, a.z * s);
}
__device__ __forceinline__ float f3_len(const float3& a) {
	return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}
__device__ __forceinline__ float f3_dot(const float3& a, const float3& b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ __forceinline__ float f3_len2(const float3& a) {
	return f3_dot(a, a);
}
__device__ __forceinline__ float3 f3_proj_perp(const float3& a, const float3& n) {
	// Project vector a to the plane perpendicular to unit direction n.
	return f3_sub(a, f3_mul(n, f3_dot(a, n)));
}

__device__ __forceinline__ float3 xpbdProjectTowardGoal(
	const float3& pi,
	const float3& goal,
	float wi,
	float dt,
	float compliance,
	float blend)
{
	const float dt2 = fmaxf(dt * dt, 1e-8f);
	const float alphaTilde = compliance / dt2;
	const float denom = wi + alphaTilde;
	if (denom < 1e-8f) {
		return pi;
	}
	const float scale = blend * (wi / denom);
	return f3_add(pi, f3_mul(f3_sub(goal, pi), scale));//XPBD 수식으로 goal 제약을 적용
}

__global__ void xpbdAngleAccumulateKernel(
	int N,
	const float3* pos_pred_in,
	const float* invMass,
	const int* offset,
	const int* nbrCount,
	const int* nbrIdx,
	const int* pairNextIdx,
	const float* restCos,
	float3* dpSum,
	int* dpCount,
	float dt,
	float angleCompliance,
	float invMassScale)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;
	if (pairNextIdx == nullptr || restCos == nullptr || dpSum == nullptr || dpCount == nullptr) return;

	const int off = offset[idx];
	const int cnt = nbrCount[idx];
	if (cnt < 2) return;

	const float3 p0 = pos_pred_in[idx];
	const float w0 = invMass[idx] * invMassScale;
	const float dt2 = fmaxf(dt * dt, 1e-8f);
	const float alpha = fmaxf(angleCompliance, 0.0f) / dt2;

	for (int k = 0; k < cnt; ++k) {
		const int e0 = off + k;
		const int i1 = nbrIdx[e0];
		const int i2 = pairNextIdx[e0];
		if (i1 < 0 || i1 >= N || i2 < 0 || i2 >= N || i1 == i2) continue;

		const float rc = restCos[e0];
		// Guard sentinel + NaN contamination explicitly.
		// (NaN fails ordered comparisons, so finite-check is required.)
		if (!isfinite(rc) || rc < -1.0f || rc > 1.0f) continue;

		const float3 p1 = pos_pred_in[i1];
		const float3 p2 = pos_pred_in[i2];
		const float3 v1 = f3_sub(p1, p0);
		const float3 v2 = f3_sub(p2, p0);
		const float l1 = f3_len(v1);
		const float l2 = f3_len(v2);
		if (l1 <= 1e-8f || l2 <= 1e-8f) continue;

		const float3 n1 = f3_mul(v1, 1.0f / l1);
		const float3 n2 = f3_mul(v2, 1.0f / l2);
		float cosCurr = f3_dot(n1, n2);
		cosCurr = fminf(1.0f, fmaxf(-1.0f, cosCurr));
		const float C = cosCurr - rc;
		if (!isfinite(C)) continue;

		const float3 g1 = f3_mul(f3_proj_perp(n2, n1), 1.0f / l1);
		const float3 g2 = f3_mul(f3_proj_perp(n1, n2), 1.0f / l2);
		const float3 g0 = f3_mul(f3_add(g1, g2), -1.0f);

		const float w1 = invMass[i1] * invMassScale;
		const float w2 = invMass[i2] * invMassScale;
		const float denom =
			w0 * f3_len2(g0) +
			w1 * f3_len2(g1) +
			w2 * f3_len2(g2);
		if (denom < 1e-8f) continue;

		float deltaLambda = -C / (denom + alpha);
		if (!isfinite(deltaLambda)) continue;
		deltaLambda = fmaxf(fminf(deltaLambda, 0.1f), -0.1f);

		float3 dp0 = f3_mul(g0, deltaLambda * w0);
		float3 dp1 = f3_mul(g1, deltaLambda * w1);
		float3 dp2 = f3_mul(g2, deltaLambda * w2);
		if (!isfinite(dp0.x) || !isfinite(dp0.y) || !isfinite(dp0.z) ||
			!isfinite(dp1.x) || !isfinite(dp1.y) || !isfinite(dp1.z) ||
			!isfinite(dp2.x) || !isfinite(dp2.y) || !isfinite(dp2.z)) {
			continue;
		}

		const float maxPairStep = fmaxf(1e-6f, 0.25f * fminf(l1, l2));
		const float len0 = f3_len(dp0);
		if (len0 > maxPairStep && len0 > 1e-8f) dp0 = f3_mul(dp0, maxPairStep / len0);
		const float len1 = f3_len(dp1);
		if (len1 > maxPairStep && len1 > 1e-8f) dp1 = f3_mul(dp1, maxPairStep / len1);
		const float len2 = f3_len(dp2);
		if (len2 > maxPairStep && len2 > 1e-8f) dp2 = f3_mul(dp2, maxPairStep / len2);

		if (w0 > 0.0f) {
			atomicAdd(&dpSum[idx].x, dp0.x);
			atomicAdd(&dpSum[idx].y, dp0.y);
			atomicAdd(&dpSum[idx].z, dp0.z);
			atomicAdd(&dpCount[idx], 1);
		}
		if (w1 > 0.0f) {
			atomicAdd(&dpSum[i1].x, dp1.x);
			atomicAdd(&dpSum[i1].y, dp1.y);
			atomicAdd(&dpSum[i1].z, dp1.z);
			atomicAdd(&dpCount[i1], 1);
		}
		if (w2 > 0.0f) {
			atomicAdd(&dpSum[i2].x, dp2.x);
			atomicAdd(&dpSum[i2].y, dp2.y);
			atomicAdd(&dpSum[i2].z, dp2.z);
			atomicAdd(&dpCount[i2], 1);
		}
	}
}

__global__ void xpbdAngleApplyKernel(
	int N,
	const float3* pos_pred_in,
	float3* pos_pred_out,
	const float* invMass,
	const float3* dpSum,
	const int* dpCount,
	float angleBlend)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	const float3 p = pos_pred_in[idx];
	if (invMass[idx] <= 0.0f || dpSum == nullptr || dpCount == nullptr) {
		pos_pred_out[idx] = p;
		return;
	}

	const int c = dpCount[idx];
	if (c <= 0) {
		pos_pred_out[idx] = p;
		return;
	}

	float3 avg = f3_mul(dpSum[idx], 1.0f / float(c));
	if (!isfinite(avg.x) || !isfinite(avg.y) || !isfinite(avg.z)) {
		pos_pred_out[idx] = p;
		return;
	}

	const float blend = fminf(fmaxf(angleBlend, 0.0f), 1.0f);
	pos_pred_out[idx] = f3_add(p, f3_mul(avg, blend));
}

__device__ __forceinline__ void mat3Identity(float m[9]) {
	m[0] = 1.0f; m[1] = 0.0f; m[2] = 0.0f;
	m[3] = 0.0f; m[4] = 1.0f; m[5] = 0.0f;
	m[6] = 0.0f; m[7] = 0.0f; m[8] = 1.0f;
}

__device__ __forceinline__ float mat3Det(const float m[9]) {
	return m[0] * (m[4] * m[8] - m[5] * m[7])
		- m[1] * (m[3] * m[8] - m[5] * m[6])
		+ m[2] * (m[3] * m[7] - m[4] * m[6]);
}

__device__ __forceinline__ void mat3Transpose(const float in[9], float out[9]) {
	out[0] = in[0]; out[1] = in[3]; out[2] = in[6];
	out[3] = in[1]; out[4] = in[4]; out[5] = in[7];
	out[6] = in[2]; out[7] = in[5]; out[8] = in[8];
}

__device__ __forceinline__ bool mat3Inverse(const float m[9], float invOut[9]) {
	const float det = mat3Det(m);
	if (fabsf(det) < 1e-8f) {
		return false;
	}
	const float invDet = 1.0f / det;
	invOut[0] = (m[4] * m[8] - m[5] * m[7]) * invDet;
	invOut[1] = (m[2] * m[7] - m[1] * m[8]) * invDet;
	invOut[2] = (m[1] * m[5] - m[2] * m[4]) * invDet;
	invOut[3] = (m[5] * m[6] - m[3] * m[8]) * invDet;
	invOut[4] = (m[0] * m[8] - m[2] * m[6]) * invDet;
	invOut[5] = (m[2] * m[3] - m[0] * m[5]) * invDet;
	invOut[6] = (m[3] * m[7] - m[4] * m[6]) * invDet;
	invOut[7] = (m[1] * m[6] - m[0] * m[7]) * invDet;
	invOut[8] = (m[0] * m[4] - m[1] * m[3]) * invDet;
	return true;
}

__device__ __forceinline__ float3 mat3MulVec(const float m[9], const float3& v) {
	return make_float3(
		m[0] * v.x + m[1] * v.y + m[2] * v.z,
		m[3] * v.x + m[4] * v.y + m[5] * v.z,
		m[6] * v.x + m[7] * v.y + m[8] * v.z);
}

__device__ __forceinline__ void polarRotation3x3(const float A[9], float R[9]) {
	for (int i = 0; i < 9; ++i) {
		R[i] = A[i];
	}

	for (int iter = 0; iter < 5; ++iter) {
		float invR[9];
		if (!mat3Inverse(R, invR)) {
			mat3Identity(R);
			return;
		}
		float invRT[9];
		mat3Transpose(invR, invRT);
		for (int i = 0; i < 9; ++i) {
			R[i] = 0.5f * (R[i] + invRT[i]);
		}
	}

	// Keep a proper rotation.
	if (mat3Det(R) < 0.0f) {
		R[2] = -R[2];
		R[5] = -R[5];
		R[8] = -R[8];
	}
}

__device__ __forceinline__ float atomicMinFloat(float* addr, float value)
{
	int* address_as_i = reinterpret_cast<int*>(addr);
	int old = *address_as_i;
	while (__int_as_float(old) > value) {
		const int assumed = old;
		old = atomicCAS(address_as_i, assumed, __float_as_int(value));
		if (assumed == old) {
			break;
		}
	}
	return __int_as_float(old);
}

struct CMConstraintGPU {
	float dx, dy, dz;
	float xShearY, xShearZ;
	float yShearX, yShearZ;
	float zShearX, zShearY;
};

__device__ __forceinline__ CMConstraintGPU cm_getConstraint(
	float density,
	float constraintGlobalScale,
	float constraintAirScale,
	float constraintSkinScale,
	float constraintBoneScale) {
	float axisC;
	float shearC;
	float materialScale;
	if (density < FORWARD::AIR) {
		axisC = constraintAirScale;
		shearC = constraintAirScale;

		//axisC = 0.3f;
		//shearC = 0.3f;
		//materialScale = constraintAirScale;
	}
	else if (density < FORWARD::SKIN) {
		axisC = constraintSkinScale;
		shearC = constraintSkinScale;

		//axisC = 0.06f;
		//shearC = 0.06f;
		//materialScale = constraintSkinScale;
	}
	else {
		axisC = constraintBoneScale;
		shearC = constraintBoneScale;

		//axisC = 0.04f;
		//shearC = 0.04f;
		//materialScale = constraintBoneScale;
	}
	const float scale = fmaxf(1e-4f, constraintGlobalScale);
	axisC *= scale;
	shearC *= scale;
	CMConstraintGPU c;
	c.dx = axisC; c.dy = axisC; c.dz = axisC;
	c.xShearY = shearC; c.xShearZ = shearC;
	c.yShearX = shearC; c.yShearZ = shearC;
	c.zShearX = shearC; c.zShearY = shearC;
	return c;
}

__device__ __forceinline__ float cm_edgeScaleFromStiff(
	float edgeStiff,
	bool useEdgeStiffness,
	float edgeStiffnessInfluence)
{
	if (!useEdgeStiffness || edgeStiffnessInfluence <= 0.0f) {
		return 1.0f;
	}
	const float s = fminf(fmaxf(edgeStiff, 0.05f), 1.0f);
	const float t = (s - 0.05f) / 0.95f;
	const float scale = 1.0f + edgeStiffnessInfluence * (t * 2.0f - 1.0f);
	return fminf(fmaxf(scale, 0.1f), 4.0f);
}

__device__ __forceinline__ float cm_propagationTime(float d0, float d1) {
	float et, nt;
	if (d0 < FORWARD::AIR) et = 1.0f;
	else if (d0 < FORWARD::SKIN) et = 0.3f;
	else if (d0 < FORWARD::BONE) et = 0.05f;
	else et = 0.005f;

	if (d1 < FORWARD::AIR) nt = 1.0f;
	else if (d1 < FORWARD::SKIN) nt = 0.3f;
	else if (d1 < FORWARD::BONE) nt = 0.05f;
	else nt = 0.005f;
	return (et + nt) * 0.5f;
}

__device__ __forceinline__ float3 cm_correctionFromNeighbor(
	const float3& epos,
	const float3& npos,
	float targDist,
	float neighborDensity,
	float edgeStiff,
	bool useEdgeStiffness,
	float edgeStiffnessInfluence,
	float constraintGlobalScale,
	float constraintAirScale,
	float constraintSkinScale,
	float constraintBoneScale)
{
	const CMConstraintGPU nC = cm_getConstraint(
		neighborDensity,
		constraintGlobalScale,
		constraintAirScale,
		constraintSkinScale,
		constraintBoneScale);
	const float edgeScale = cm_edgeScaleFromStiff(edgeStiff, useEdgeStiffness, edgeStiffnessInfluence);
	const float effectiveDx = nC.dx;// / edgeScale;
	float3 dir = f3_sub(epos, npos);
	float len = f3_len(dir);
	if (len < 1e-6f) return make_float3(0.0f, 0.0f, 0.0f);
	float invLen = 1.0f / len;
	float3 nDir = f3_mul(dir, invLen);

	if (len < targDist - effectiveDx) {//너무 가까울때
		float delta = (targDist - effectiveDx) - len;// 임계보다 가까운만큼의 차이를 delta로
		return f3_mul(nDir, delta * edgeScale); // push away
	}
	else if (len > targDist + effectiveDx) {// 너무 멀때
		float delta = len - (targDist + effectiveDx);
		return f3_mul(nDir, -(delta * edgeScale)); // pull closer
	}
	return make_float3(0.0f, 0.0f, 0.0f);
}

__global__ void chainmailPropagateKernel(
	int N,
	const float3* pos_in,
	float3* pos_out,
	const float* time_in,
	float* time_out,
	const float* density,
	const int* offset,
	const int* nbrCount,
	const int* nbrIdx,
	const float* nbrDist,
	const float* nbrStiff,
	float propStrength,
	float constraintGlobalScale,
	float constraintAirScale,
	float constraintSkinScale,
	float constraintBoneScale,
	bool useEdgeStiffness,
	float edgeStiffnessInfluence)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	const float3 epos = pos_in[idx];
	const float edens = density[idx];
	const float tcur = time_in[idx];

	float bestTime = tcur;
	float3 accum = make_float3(0.0f, 0.0f, 0.0f);
	int corrCount = 0;

	const int off = offset[idx];
	const int cnt = nbrCount[idx];
	for (int j = 0; j < cnt; ++j) {
		const int nIdx = nbrIdx[off + j];
		const float ntime = time_in[nIdx];
		const float ndens = density[nIdx];
		const float cand = ntime + cm_propagationTime(edens, ndens);// 자신과 이웃의 density 를 넣음
		if (cand < bestTime) bestTime = cand;

		if (cand < tcur) {
			const float3 npos = pos_in[nIdx];
			const float targDist = nbrDist[off + j];
			const float edgeStiff = nbrStiff ? nbrStiff[off + j] : 1.0f;
			float3 corr = cm_correctionFromNeighbor(
				epos, npos, targDist, ndens, edgeStiff, useEdgeStiffness,
				edgeStiffnessInfluence, constraintGlobalScale, constraintAirScale,
				constraintSkinScale, constraintBoneScale);
			accum = f3_add(accum, corr);
			corrCount++;
		}
	}

	float3 outPos = epos;
	if (corrCount > 0) {
		float3 avgCorr = f3_mul(accum, 1.0f / float(corrCount));//이웃들로부터 받은 보정량을 평균해서 과도한 이동을 막기위한 안정화. 이웃들이 많거나 적으면 차이가 있을수있기때문.
		outPos = f3_add(epos, f3_mul(avgCorr, propStrength));
	}

	pos_out[idx] = outPos;
	time_out[idx] = bestTime;
}

// HP-ChainMail style: pick the single fastest neighbor (smallest cand) and move only w.r.t. that neighbor.
__global__ void chainmailPropagateKernelBest(
	int N,
	const float3* pos_in,
	float3* pos_out,
	const float* time_in,
	float* time_out,
	const float* density,
	const int* offset,
	const int* nbrCount,
	const int* nbrIdx,
	const float* nbrDist,
	const float* nbrStiff,
	float propStrength,
	float constraintGlobalScale,
	float constraintAirScale,
	float constraintSkinScale,
	float constraintBoneScale,
	bool useEdgeStiffness,
	float edgeStiffnessInfluence)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	const float3 epos = pos_in[idx];
	const float edens = density[idx];
	const float tcur = time_in[idx];

	float bestTime = tcur;
	int bestIdx = -1;
	float bestDist = 0.0f;
	float bestNDens = 0.0f;

	const int off = offset[idx];
	const int cnt = nbrCount[idx];
	for (int j = 0; j < cnt; ++j) {
		const int nIdx = nbrIdx[off + j];
		const float ntime = time_in[nIdx];
		const float ndens = density[nIdx];
		const float cand = ntime + cm_propagationTime(edens, ndens);
		if (cand < bestTime) {
			bestTime = cand;
			bestIdx = nIdx;
			bestDist = nbrDist[off + j];
			bestNDens = ndens;
		}
	}

	float3 outPos = epos;
	if (bestIdx >= 0 && bestTime < tcur) {
		const float3 npos = pos_in[bestIdx];
		float bestEdgeStiff = 1.0f;
		for (int j = 0; j < cnt; ++j) {
			if (nbrIdx[off + j] == bestIdx) {
				bestEdgeStiff = nbrStiff ? nbrStiff[off + j] : 1.0f;
				break;
			}
		}
		const float3 corr = cm_correctionFromNeighbor(
			epos, npos, bestDist, bestNDens, bestEdgeStiff, useEdgeStiffness,
			edgeStiffnessInfluence, constraintGlobalScale, constraintAirScale,
			constraintSkinScale, constraintBoneScale);
		if (f3_len(corr) > 0.0f) {
			outPos = f3_add(epos, f3_mul(corr, propStrength));
		}
	}

	pos_out[idx] = outPos;
	time_out[idx] = bestTime;
}

__global__ void chainmailRelaxKernel(
	int N,
	const float3* pos_in,
	float3* pos_out,
	const float* density,
	const int* offset,
	const int* nbrCount,
	const int* nbrIdx,
	const float* nbrDist,
	const float* nbrStiff,
	float stiffness,
	float damping,
	float constraintGlobalScale,
	float constraintAirScale,
	float constraintSkinScale,
	float constraintBoneScale,
	bool useEdgeStiffness,
	float edgeStiffnessInfluence)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	const float3 epos = pos_in[idx];
	const float edens = density[idx];
	const CMConstraintGPU eC = cm_getConstraint(
		edens, constraintGlobalScale, constraintAirScale, constraintSkinScale, constraintBoneScale);

	float3 accum = make_float3(0.0f, 0.0f, 0.0f);
	int corrCount = 0;
	float sumTarget = 0.0f;
	int distCount = 0;

	const int off = offset[idx];
	const int cnt = nbrCount[idx];
	for (int j = 0; j < cnt; ++j) {
		const int nIdx = nbrIdx[off + j];
		const float3 npos = pos_in[nIdx];
		const float ndens = density[nIdx];
		const CMConstraintGPU nC = cm_getConstraint(
			ndens, constraintGlobalScale, constraintAirScale, constraintSkinScale, constraintBoneScale);
		const float edgeStiff = nbrStiff ? nbrStiff[off + j] : 1.0f;
		const float edgeScale = cm_edgeScaleFromStiff(edgeStiff, useEdgeStiffness, edgeStiffnessInfluence);

		const float targDist = nbrDist[off + j];
		// Relax tolerance directly controls "hard/soft" perception.
		// Previous factor (0.002x) made UI scaling visually subtle.
		const float relaxTolScale = 0.0000001f;
		const float tol = (relaxTolScale * (eC.dx + nC.dx)) / edgeScale;

		float3 dir = f3_sub(npos, epos);
		float len = f3_len(dir);
		if (len < 1e-6f) continue;

		float invLen = 1.0f / len;
		float3 nDir = f3_mul(dir, invLen);

		if (len < targDist - tol) {
			float delta = (targDist - tol) - len;
			accum = f3_sub(accum, f3_mul(nDir, delta * edgeScale));
			corrCount++;
		}
		else if (len > targDist + tol) {
			float delta = len - (targDist + tol);
			accum = f3_add(accum, f3_mul(nDir, delta * edgeScale));
			corrCount++;
		}

		sumTarget += targDist;
		distCount++;
	}

	if (corrCount > 0) {
		float3 avgCorr = f3_mul(accum, 1.0f / float(corrCount));

		float maxStep = 0.6f;
		if (distCount > 0) {
			float avgTarget = sumTarget / float(distCount);
			const float maxStepFactor = 1.5f;
			maxStep = maxStepFactor * avgTarget;
		}

		float corrLen = f3_len(avgCorr);
		if (corrLen > 1e-6f && corrLen > maxStep) {
			avgCorr = f3_mul(avgCorr, maxStep / corrLen);
		}

		float3 corrected = f3_add(epos, f3_mul(avgCorr, stiffness));
		// damping: reduce correction magnitude (0 = no damping, 1 = fully frozen)
		pos_out[idx] = f3_add(epos, f3_mul(f3_sub(corrected, epos), 1.0f - damping));
	}
	else {
		pos_out[idx] = epos;
	}
}

__global__ void chainmailPropagateKernelActive(
	int N,
	const float3* pos_in,
	float3* pos_out,
	const float* time_in,
	float* time_out,
	const float* density,
	const int* offset,
	const int* nbrCount,
	const int* nbrIdx,
	const float* nbrDist,
	const float* nbrStiff,
	float propStrength,
	float constraintGlobalScale,
	float constraintAirScale,
	float constraintSkinScale,
	float constraintBoneScale,
	bool useEdgeStiffness,
	float edgeStiffnessInfluence,
	const int* active_map,
	int* next_map)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	if (!active_map || active_map[idx] == 0) {
		pos_out[idx] = pos_in[idx];
		time_out[idx] = time_in[idx];
		return;
	}

	const float3 epos = pos_in[idx];
	const float edens = density[idx];
	const float tcur = time_in[idx];

	float bestTime = tcur;
	float3 accum = make_float3(0.0f, 0.0f, 0.0f);
	int corrCount = 0;

	const int off = offset[idx];
	const int cnt = nbrCount[idx];
	for (int j = 0; j < cnt; ++j) {
		const int nIdx = nbrIdx[off + j];
		const float ntime = time_in[nIdx];
		const float ndens = density[nIdx];
		const float propTime = cm_propagationTime(edens, ndens);

		const float cand = ntime + propTime;
		if (cand < bestTime) bestTime = cand;

		const float3 npos = pos_in[nIdx];
		const float targDist = nbrDist[off + j];
		const float edgeStiff = nbrStiff ? nbrStiff[off + j] : 1.0f;
		float3 corr = cm_correctionFromNeighbor(
			epos, npos, targDist, ndens, edgeStiff, useEdgeStiffness,
			edgeStiffnessInfluence, constraintGlobalScale, constraintAirScale,
			constraintSkinScale, constraintBoneScale);
		if (f3_len(corr) > 0.0f) {
			accum = f3_add(accum, corr);
			corrCount++;
			// Spread activity to neighbors that actually need correction.
			next_map[nIdx] = 1;
		}
	}

	const float baseTime = (bestTime < tcur) ? bestTime : tcur;
	for (int j = 0; j < cnt; ++j) {
		const int nIdx = nbrIdx[off + j];
		const float ntime = time_in[nIdx];
		const float ndens = density[nIdx];
		const float propTime = cm_propagationTime(edens, ndens);
		if (baseTime + propTime < ntime) {
			next_map[nIdx] = 1;
		}
	}

	float3 outPos = epos;
	if (corrCount > 0) {
		float3 avgCorr = f3_mul(accum, 1.0f / float(corrCount));
		outPos = f3_add(epos, f3_mul(avgCorr, propStrength));
	}

	pos_out[idx] = outPos;
	time_out[idx] = bestTime;

	if (corrCount > 0 || bestTime < tcur) {
		next_map[idx] = 1;
	}
}

__global__ void chainmailPropagateScatterKernel(
	int N,
	const float* time_in,
	float* time_out,
	const float* density,
	const int* offset,
	const int* nbrCount,
	const int* nbrIdx,
	const int* active_map,
	int* best_from,
	int* next_map)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;
	if (!active_map || active_map[idx] == 0) return;

	const float tcur = time_in[idx];
	const float edens = density[idx];
	const int off = offset[idx];
	const int cnt = nbrCount[idx];

	next_map[idx] = 1;

	for (int j = 0; j < cnt; ++j) {
		const int nIdx = nbrIdx[off + j];
		const float ndens = density[nIdx];
		const float newTime = tcur + cm_propagationTime(edens, ndens);
		const float old = atomicMinFloat(time_out + nIdx, newTime);
		if (newTime < old) {
			best_from[nIdx] = idx;
			next_map[nIdx] = 1;
		}
	}
}

__global__ void applyBestFromKernel(
	int N,
	const float3* pos_in,
	float3* pos_out,
	const float* density,
	const int* offset,
	const int* nbrCount,
	const int* nbrIdx,
	const float* nbrDist,
	const float* nbrStiff,
	float constraintGlobalScale,
	float constraintAirScale,
	float constraintSkinScale,
	float constraintBoneScale,
	bool useEdgeStiffness,
	float edgeStiffnessInfluence,
	const int* best_from,
	int* next_map)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	const int src = best_from[idx];
	if (src < 0) {
		pos_out[idx] = pos_in[idx];
		return;
	}

	float targStiff = 1.0f;
	float targDist = -1.0f;
	const int offSrc = offset[src];
	const int cntSrc = nbrCount[src];
	for (int j = 0; j < cntSrc; ++j) {
		if (nbrIdx[offSrc + j] == idx) {
			targDist = nbrDist[offSrc + j];
			targStiff = nbrStiff ? nbrStiff[offSrc + j] : 1.0f;
			break;
		}
	}
	if (targDist < 0.0f) {
		const int off = offset[idx];
		const int cnt = nbrCount[idx];
		for (int j = 0; j < cnt; ++j) {
			if (nbrIdx[off + j] == src) {
				targDist = nbrDist[off + j];
				targStiff = nbrStiff ? nbrStiff[off + j] : 1.0f;
				break;
			}
		}
	}
	if (targDist < 0.0f) {
		pos_out[idx] = pos_in[idx];
		return;
	}

	const float3 epos = pos_in[idx];
	const float3 npos = pos_in[src];
	const float ndens = density[src];
	const float3 corr = cm_correctionFromNeighbor(
		epos, npos, targDist, ndens, targStiff, useEdgeStiffness,
		edgeStiffnessInfluence, constraintGlobalScale, constraintAirScale,
		constraintSkinScale, constraintBoneScale);
	if (f3_len(corr) > 0.0f) {
		pos_out[idx] = f3_add(epos, corr);
		next_map[idx] = 1;
	}
	else {
		pos_out[idx] = epos;
	}
}

__global__ void chainmailRelaxKernelActive(
	int N,
	const float3* pos_in,
	float3* pos_out,
	const float* density,
	const int* offset,
	const int* nbrCount,
	const int* nbrIdx,
	const float* nbrDist,
	const float* nbrStiff,
	float stiffness,
	float damping,
	float constraintGlobalScale,
	float constraintAirScale,
	float constraintSkinScale,
	float constraintBoneScale,
	bool useEdgeStiffness,
	float edgeStiffnessInfluence,
	const int* active_map,
	int* next_map)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	if (!active_map || active_map[idx] == 0) {
		pos_out[idx] = pos_in[idx];
		return;
	}

	const float3 epos = pos_in[idx];
	const float edens = density[idx];
	const CMConstraintGPU eC = cm_getConstraint(
		edens, constraintGlobalScale, constraintAirScale, constraintSkinScale, constraintBoneScale);

	float3 accum = make_float3(0.0f, 0.0f, 0.0f);
	int corrCount = 0;
	float sumTarget = 0.0f;
	int distCount = 0;

	const int off = offset[idx];
	const int cnt = nbrCount[idx];
	for (int j = 0; j < cnt; ++j) {
		const int nIdx = nbrIdx[off + j];
		const float3 npos = pos_in[nIdx];
		const float ndens = density[nIdx];
		const CMConstraintGPU nC = cm_getConstraint(
			ndens, constraintGlobalScale, constraintAirScale, constraintSkinScale, constraintBoneScale);
		const float edgeStiff = nbrStiff ? nbrStiff[off + j] : 1.0f;
		const float edgeScale = cm_edgeScaleFromStiff(edgeStiff, useEdgeStiffness, edgeStiffnessInfluence);

		const float targDist = nbrDist[off + j];
		// Relax tolerance directly controls "hard/soft" perception.
		// Previous factor (0.002x) made UI scaling visually subtle.
		const float relaxTolScale = 0.10f;
		const float tol = (relaxTolScale * (eC.dx + nC.dx)) / edgeScale;

		float3 dir = f3_sub(npos, epos);
		float len = f3_len(dir);
		if (len < 1e-6f) continue;

		float invLen = 1.0f / len;
		float3 nDir = f3_mul(dir, invLen);

		if (len < targDist - tol) {
			float delta = (targDist - tol) - len;
			accum = f3_sub(accum, f3_mul(nDir, delta * edgeScale));
			corrCount++;
		}
		else if (len > targDist + tol) {
			float delta = len - (targDist + tol);
			accum = f3_add(accum, f3_mul(nDir, delta * edgeScale));
			corrCount++;
		}

		sumTarget += targDist;
		distCount++;
	}

	if (corrCount > 0) {
		float3 avgCorr = f3_mul(accum, 1.0f / float(corrCount));

		float maxStep = 0.6f;
		if (distCount > 0) {
			float avgTarget = sumTarget / float(distCount);
			const float maxStepFactor = 1.5f;
			maxStep = maxStepFactor * avgTarget;
		}

		float corrLen = f3_len(avgCorr);
		if (corrLen > 1e-6f && corrLen > maxStep) {
			avgCorr = f3_mul(avgCorr, maxStep / corrLen);
		}

		float3 corrected = f3_add(epos, f3_mul(avgCorr, stiffness));
		pos_out[idx] = f3_add(epos, f3_mul(f3_sub(corrected, epos), 1.0f - damping));
		next_map[idx] = 1;
	}
	else {
		pos_out[idx] = epos;
	}
}

__global__ void chainmailInertiaKernel(
	int N,
	const float3* pos_prev,
	float3* pos_curr,
	float3* vel,
	const float* invMass,
	float inertiaGain,// 총 계산된 관성 반영 파라메터
	float velocityRetention,//에너지 반영 파라메터(낮을수록 제동걸림)
	float velocityClamp//충격량 파라메터(낮을수록 적용되는 속도 제한)
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	if (invMass && invMass[idx] <= 0.0f) {
		vel[idx] = make_float3(0.0f, 0.0f, 0.0f);
		return;
	}

	const float3 disp = f3_sub(pos_curr[idx], pos_prev[idx]);//이번 프레임에서 체인메일 알고리즘(전파/안정화)에 의해 가우시안이 실제로 이동한 거리와 방향
	float3 v = f3_add(f3_mul(vel[idx], velocityRetention), disp);//이전 프레임에서 가지고 있던 속도(vel)를 일정 비율 유지하고, 여기에 이번에 새로 발생한 움직임(disp)을 더함
	const float vLen = f3_len(v);//이를 통해 가우시안은 "방금 움직였던 방향으로 계속 가려는 성질"을 기억
	if (!isfinite(v.x) || !isfinite(v.y) || !isfinite(v.z)) {
		v = make_float3(0.0f, 0.0f, 0.0f);
	}
	else if (vLen > velocityClamp && vLen > 1e-8f) {
		v = f3_mul(v, velocityClamp / vLen);
	}
	vel[idx] = v;
	pos_curr[idx] = f3_add(pos_curr[idx], f3_mul(v, inertiaGain));
}

__global__ void countActiveKernel(int N, const int* active_map, int* out_count)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;
	if (active_map && active_map[idx]) {
		atomicAdd(out_count, 1);
	}
}

__global__ void resetTimeKernel(int N, float* time, float value)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;
	time[idx] = value;
}

__global__ void applySeedCommandsKernel(
	int n,
	const int* idx,
	const float3* delta,
	float3* pos,
	float* time,
	int* active_map)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	const int id = idx[i];
	const float3 d = delta[i];
	pos[id] = f3_add(pos[id], d);
	time[id] = 0.0f;
	if (active_map) {
		active_map[id] = 1;
	}
}

__global__ void xpbdResetInvMassKernel(int N, float* invMass, float value)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;
	invMass[idx] = value;
}

__global__ void applySeedCommandsKernel(
	int n,
	const int* idx,
	const float3* delta,
	float3* pos,
	float3* vel,
	float* invMass,
	float* time)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	const int id = idx[i];
	const float3 d = delta[i];
	atomicAdd(&pos[id].x, d.x);
	atomicAdd(&pos[id].y, d.y);
	atomicAdd(&pos[id].z, d.z);
	vel[id] = make_float3(0.0f, 0.0f, 0.0f);
	invMass[id] = 0.0f; // kinematic pin
	time[id] = 0.0f;
}

__global__ void xpbdPredictKernel(
	int N,
	const float3* pos_curr,
	float3* pos_pred,
	float3* vel,
	const float* invMass,
	float dt,
	float3 gravity)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	float3 x = pos_curr[idx];//기존 위치
	float3 v = vel[idx];//기존가속도
	const float w = invMass[idx];//기존 질량 // 역질량 즉 0이면 가장 무거운것

	if (w > 0.0f) {//일반적인 질량을 가진 가우시안 관성과 중력에 영향을 받자!!
		v = f3_add(v, f3_mul(gravity, dt));//질량에 중력 적용하여 가속도 계산
		pos_pred[idx] = f3_add(x, f3_mul(v, dt));//기존 위치에 적용해 예측된 위치 계산
		vel[idx] = v;//계산된 가속도 
	}
	else {//가장 무거운것이라면(질량무한대) 아무리강한 힘에도 움직이지않는 고정
		pos_pred[idx] = x;//움직이지 않아야하기에 기존위치를 예측된위치로 그냥 넘김
		vel[idx] = make_float3(0.0f, 0.0f, 0.0f);//가속도 또한 0
	}
}

// ── 형상 제약 회전 추출 (double + 상대 멈춤 기준) ────────────────────────────
// 기존 float32 경로는 AᵀA 로 조건수가 제곱되는데 Jacobi 멈춤이 절대값 1e-6 이라, 가늘고 긴(막대형) 덩어리에서
// 작은 두 고유벡터가 덜 풀린 채 멈춰 '가만히 있어도' 축 둘레로 틀린 회전을 냈다. 매 반복 같은 방향이라
// 줄기·머리카락·안경테가 제자리에서 계속 돈다. (외부 검토 지적 → 독립 재현: scratchpad double_polar_check.py)
//   검토 픽스처(반축 √[1,1e-3,1e-5], 변형 없음): 기존 29.7°/pass 누적 → 0°
//   정지 상태 목표 위치 오차(덩어리 크기 대비): 막대 0.03 기존 0.0092 → 0
//   180° 강체 회전 추적: 막대 0.01 기존 1.0(실패) → 0, 나머지 모양도 0
//   가는 가닥에 1-ring 형상만, 외력 0: 기존 60프레임에 물체 크기의 1.5% 표류 → 0
// ※ '막대형이면 축 방향만 맞추기'(비틀림 0)도 시험했으나 회전 추적 오차 0.13~0.46 으로 스핀·구르기와 충돌해 기각.
// 알고리즘은 기존과 같다(AᵀA Jacobi → U = A V S⁻¹ → 납작 보정 → 반사 방지). 입력 A 는 Frobenius 정규화된 glm 행렬.
__device__ inline bool shapeMatchRotationDouble(const glm::mat3& A, glm::mat3& Rout)
{
	double a[3][3];                         // 행우선 a[r][c] (glm 은 열우선 A[c][r])
	for (int r = 0; r < 3; ++r)
		for (int c = 0; c < 3; ++c)
			a[r][c] = (double)A[c][r];

	double m[3][3];                         // AᵀA
	for (int r = 0; r < 3; ++r)
		for (int c = 0; c < 3; ++c)
			m[r][c] = a[0][r] * a[0][c] + a[1][r] * a[1][c] + a[2][r] * a[2][c];

	double v[3][3] = { { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 0.0, 0.0, 1.0 } };
	const int P[3] = { 0, 0, 1 };
	const int Q[3] = { 1, 2, 2 };
	for (int iter = 0; iter < 50; ++iter) {
		// 모든 비대각 원소가 각자의 대각 규모 대비 작으면 종료 — 씬 스케일·이방성과 무관한 상대 기준
		bool done = true;
		int p = 0, q = 1;
		double maxOff = -1.0;
		for (int k = 0; k < 3; ++k) {
			const double off = fabs(m[P[k]][Q[k]]);
			if (off > 1e-12 * sqrt(fabs(m[P[k]][P[k]] * m[Q[k]][Q[k]])) + 1e-300) done = false;
			if (off > maxOff) { maxOff = off; p = P[k]; q = Q[k]; }
		}
		if (done) break;
		const double app = m[p][p], aqq = m[q][q], apq = m[p][q];
		const double phi = 0.5 * (aqq - app) / (apq + 1e-300);
		const double t = (phi >= 0.0) ? 1.0 / (phi + sqrt(1.0 + phi * phi)) : 1.0 / (phi - sqrt(1.0 + phi * phi));
		const double c = 1.0 / sqrt(1.0 + t * t);
		const double s = t * c;
		m[p][p] = app - t * apq;
		m[q][q] = aqq + t * apq;
		m[p][q] = m[q][p] = 0.0;
		for (int r = 0; r < 3; ++r) {
			if (r == p || r == q) continue;
			const double arp = m[r][p], arq = m[r][q];
			m[r][p] = m[p][r] = c * arp - s * arq;
			m[r][q] = m[q][r] = c * arq + s * arp;
		}
		for (int r = 0; r < 3; ++r) {
			const double vrp = v[r][p], vrq = v[r][q];
			v[r][p] = c * vrp - s * vrq;
			v[r][q] = s * vrp + c * vrq;
		}
	}

	// 고유값 오름차순 정렬 (고유벡터 열도 함께 교환)
	double ev[3] = { m[0][0], m[1][1], m[2][2] };
	for (int i = 0; i < 2; ++i) {
		for (int j = i + 1; j < 3; ++j) {
			if (ev[i] > ev[j]) {
				const double te = ev[i]; ev[i] = ev[j]; ev[j] = te;
				for (int r = 0; r < 3; ++r) { const double tv = v[r][i]; v[r][i] = v[r][j]; v[r][j] = tv; }
			}
		}
	}

	double S[3], Sinv[3];
	for (int i = 0; i < 3; ++i) S[i] = sqrt(ev[i] > 0.0 ? ev[i] : 0.0);
	const double Smax = (S[2] > 1e-300) ? S[2] : 1e-300;
	for (int i = 0; i < 3; ++i) Sinv[i] = (S[i] > 1e-12 * Smax) ? 1.0 / S[i] : 0.0;   // 컷오프도 상대값

	double u[3][3];                         // U = A V S⁻¹ (열 c 가 u_c)
	for (int r = 0; r < 3; ++r)
		for (int c = 0; c < 3; ++c)
			u[r][c] = (a[r][0] * v[0][c] + a[r][1] * v[1][c] + a[r][2] * v[2][c]) * Sinv[c];

	if (S[0] < 0.08 * Smax) {               // 납작/막대: 가장 작은 축은 나머지 두 축의 외적으로
		const double cx = u[1][1] * u[2][2] - u[2][1] * u[1][2];
		const double cy = u[2][1] * u[0][2] - u[0][1] * u[2][2];
		const double cz = u[0][1] * u[1][2] - u[1][1] * u[0][2];
		const double len = sqrt(cx * cx + cy * cy + cz * cz);
		if (!(len > 1e-7)) return false;     // 두 축도 축퇴 → 기존 경로처럼 포기
		u[0][0] = cx / len; u[1][0] = cy / len; u[2][0] = cz / len;
	}

	double R[3][3];
	for (int pass = 0; pass < 2; ++pass) {
		for (int r = 0; r < 3; ++r)
			for (int c = 0; c < 3; ++c)
				R[r][c] = u[r][0] * v[c][0] + u[r][1] * v[c][1] + u[r][2] * v[c][2];   // R = U Vᵀ
		const double det = R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1])
			- R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0])
			+ R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]);
		if (det >= 0.0 || pass == 1) break;
		for (int r = 0; r < 3; ++r) u[r][0] = -u[r][0];   // 반사 방지: 가장 작은 축 반전 후 재조립
	}
	for (int r = 0; r < 3; ++r) {
		for (int c = 0; c < 3; ++c) {
			if (!isfinite(R[r][c])) return false;
			Rout[c][r] = (float)R[r][c];
		}
	}
	return true;
}

__global__ void xpbdShapeMatchingKernel(
	int N,
	const float3* pos_pred_in,
	float3* pos_pred_out,
	const float3* pos_rest,
	const float* invMass,
	const int* offset,
	const int* nbrCount,
	const int* nbrIdx,
	float dt,
	float invMassScale,
	float shapeCompliance,
	float shapeBlend,
	int robustPolar)   // 1 = shapeMatchRotationDouble, 0 = 기존 float32 경로
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	const float3 pi = pos_pred_in[idx];//현재 예측되는위치
	const float wi = invMass[idx] * invMassScale;//역질량

	//질량 무한대(고정점) 이면 통과 즉 마우스로 잡은점
	if (wi <= 0.0f || pos_rest == nullptr) {
		pos_pred_out[idx] = pi;
		return;
	}

	const int off = offset[idx];
	const int cnt = nbrCount[idx];//이웃개수
	if (cnt < 4) {//3x3 행렬을 안정적으로 구성하기위해 span 하는 이웃 최소 3개이상필요 적을시 기존위치 반환후 포기
		// Degenerate local cluster: avoid injecting global-rest tether force.
		pos_pred_out[idx] = pi;
		return;
	}
	// 나와 이웃들의 무게중심 구하기
	float3 comRest = pos_rest[idx];// 나자신 기존 위치
	float3 comCurr = pi;//현재 예측되는위치(매프레임 변화)
	float clusterCount = 1.0f;//자기 자신포함해서 시작

	//이 무게중심이 필요한 이유는 A 행렬을 구성할 때 절대 위치가 아닌 무게중심 기준 상대 위치로 계산해야 하기 때문.
	//절대 위치로 계산하면 이동(translation)이 회전으로 잘못 해석됨
	for (int k = 0; k < cnt; ++k) {// 볼륨에서 사면체를 한 단위로 하듯이 한 이웃뭉치를 한단위로 지정
		const int nIdx = nbrIdx[off + k];//이웃 인덱스
		if (nIdx < 0 || nIdx >= N) continue;
		comRest = f3_add(comRest, pos_rest[nIdx]);//이웃들의 기존위치와 나자신 기존위치 모두 더함
		comCurr = f3_add(comCurr, pos_pred_in[nIdx]);//현재 변형되는  이웃 , 나 자신 위치 모두 더함
		clusterCount += 1.0f;
	}

	if (clusterCount < 2.0f) {
		// Not enough valid neighbors for stable shape matching.
		pos_pred_out[idx] = pi;
		return;
	}
	//무게중심 구함 기존위치, 변형되는 위치 기준으로
	const float invCluster = 1.0f / clusterCount;
	comRest = f3_mul(comRest, invCluster);
	comCurr = f3_mul(comCurr, invCluster);

	// [FIX 2] 경계/낭떨어지 가우시안 감지
// 이웃들이 한쪽으로 쏠려있으면 A 행렬이 편향됨 → shape matching 포기
	{
		// 무게중심에서 각 이웃까지의 방향 벡터 합산
		float3 dirSum = make_float3(0.0f, 0.0f, 0.0f);
		float maxLen = 0.0f;

		for (int k = 0; k < cnt; ++k) {
			const int nIdx = nbrIdx[off + k];
			if (nIdx < 0 || nIdx >= N) continue;
			const float3 q_f3 = f3_sub(pos_rest[nIdx], comRest);
			const float len = f3_len(q_f3);
			if (len > 1e-7f) {
				// 정규화해서 방향만 합산
				dirSum = f3_add(dirSum, f3_mul(q_f3, 1.0f / len));
				if (len > maxLen) maxLen = len;
			}
		}

		// 방향 벡터 합의 크기 / 이웃 수 = 편향도
		// 0이면 완벽하게 균등, 1에 가까우면 완전히 한쪽으로 쏠림
		const float bias = f3_len(dirSum) / fmaxf((float)cnt, 1.0f);

		// 편향도가 0.6 이상이면 경계 가우시안으로 판단 → shape matching 포기
		const float bias_threshold = 0.4f;
		if (bias > bias_threshold) {
			pos_pred_out[idx] = pi;
			return;
		}
	}
	// 여기까지 
	// 
	

	// Eigen 대신 glm으로 A 행렬 조립
	glm::mat3 A(0.0f);
	// 1. [복구됨] 나 자신의 상대 좌표 외적 누적
	{
		const float3 q_f3 = f3_sub(pos_rest[idx], comRest);//기존  rest 상태에서 무게중심 기준 상대벡터
		const float3 p_f3 = f3_sub(pi, comCurr);//현재 상태에서 무게중심 기준 상대벡터
		glm::vec3 q(q_f3.x, q_f3.y, q_f3.z);
		glm::vec3 p(p_f3.x, p_f3.y, p_f3.z);
		A += glm::outerProduct(p, q);
	}
	// 2. 나와 이웃들의 외적 누적
	for (int k = 0; k < cnt; ++k) {
		const int nIdx = nbrIdx[off + k];
		if (nIdx < 0 || nIdx >= N) continue;
		const float3 q_f3 = f3_sub(pos_rest[nIdx], comRest);//기존  rest 상태에서 무게중심 기준 상대벡터
		const float3 p_f3 = f3_sub(pos_pred_in[nIdx], comCurr);//현재 상태에서 무게중심 기준 상대벡터

		// glm::vec3로 변환
		glm::vec3 q(q_f3.x, q_f3.y, q_f3.z);
		glm::vec3 p(p_f3.x, p_f3.y, p_f3.z);

		// A += p * q^T (외적)
		A += glm::outerProduct(p, q);
	}
	//이 클러스터가 rest에서 현재까지 어떻게 움직였는가 를 담은 3×3 변형 기록
	// 
	// 3. Float 언더플로우(노이즈 붕괴) 방지를 위한 A 정규화
	//A를 Frobenius norm으로 나누어 스케일을 1로 정규화. 
	//AtA 고유값 분해에서 float32 정밀도 문제를 줄이기 위해서
	//이 정규화는 R 추출에 영향을 주지 않음.
	//A와 (A/a_norm)은 같은 방향 정보를 가지므로 Polar Decomposition 결과 R은 동일. 스케일만 제거하고 회전 정보는 그대로 보존
	float a_norm_sq = 0.0f;
	for (int c = 0; c < 3; ++c) {
		for (int r = 0; r < 3; ++r) {
			a_norm_sq += A[c][r] * A[c][r];
		}
	}
	float a_norm = sqrtf(a_norm_sq);

	// 부피가 아예 없는 노이즈이거나 완벽한 평면인 경우 회전 포기
	//a_norm이 1e-8 미만이면 클러스터가 완전히 뭉쳐있거나 모든 점이 무게중심에 집중된 경우로, 변형 정보가 없으므로 포기
	if (!isfinite(a_norm) || a_norm < 1e-8f) {
		// Near-singular covariance; skip unreliable correction.
		pos_pred_out[idx] = pi;
		return;
	}

	// A를 1.0 스케일로 뻥튀기! (회전 U, V는 불변하고 Float 정밀도만 극대화됨)
	A /= a_norm;

	glm::mat3 R(1.0f);
	if (robustPolar) {
		if (!shapeMatchRotationDouble(A, R)) {
			pos_pred_out[idx] = pi;
			return;
		}
	}
	else {
	// ── 기존 float32 경로 (A/B 비교용으로 그대로 보존. 막대형 덩어리에서 정지 상태에도 틀린 회전을 낸다) ──

	// 1. 대칭 행렬 (A^T * A) 만들기 SVD를 통한 Polar Decomposition
	glm::mat3 AtA = glm::transpose(A) * A;

	// 2. 야코비 고유값 분해 호출
	glm::vec3 S_squared; // 고유값 (스케일의 제곱)
	glm::mat3 V;         // 고유벡터 (V 행렬)
	eigenDecomposition_glm(AtA, S_squared, V);

	// 3. S_vec (특이값 = 스케일) 구하기 및 0 나누기 방지
	glm::vec3 S_vec = glm::sqrt(glm::max(glm::vec3(0.0f), S_squared));
	if (!isfinite(S_vec.x) || !isfinite(S_vec.y) || !isfinite(S_vec.z)) {
		pos_pred_out[idx] = pi;
		return;
	}

	const float epsilon = 1e-7f;
	glm::vec3 S_inv_vec = glm::vec3(
		(S_vec.x > epsilon) ? 1.0f / S_vec.x : 0.0f,
		(S_vec.y > epsilon) ? 1.0f / S_vec.y : 0.0f,
		(S_vec.z > epsilon) ? 1.0f / S_vec.z : 0.0f
	);

	// 4. U 행렬 계산 (U = A * V * S^-1)
	glm::mat3 S_inv_mat = glm::mat3(
		S_inv_vec.x, 0.0f, 0.0f,
		0.0f, S_inv_vec.y, 0.0f,
		0.0f, 0.0f, S_inv_vec.z
	);
	glm::mat3 U = A * V * S_inv_mat;
	if (!isfinite(U[0][0]) || !isfinite(U[0][1]) || !isfinite(U[0][2]) ||
		!isfinite(U[1][0]) || !isfinite(U[1][1]) || !isfinite(U[1][2]) ||
		!isfinite(U[2][0]) || !isfinite(U[2][1]) || !isfinite(U[2][2])) {
		pos_pred_out[idx] = pi;
		return;
	}


	// 예외 처리 추가한 부분
	const float flat_eps = 0.08f;
	if (S_vec.x < flat_eps * fmaxf(S_vec.z, 1e-7f)) {
		glm::vec3 col0 = glm::cross(U[1], U[2]);
		float col0_len = glm::length(col0);
		if (col0_len > 1e-7f) {
			U[0] = col0 / col0_len;
		}
		else {
			// U[1], U[2]도 축퇴 → shape matching 포기
			pos_pred_out[idx] = pi;
			return;
		}
	}

	///////
	// 

	// 5. 순수 회전 행렬 R 추출 (R = U * V^T)
	R = U * glm::transpose(V);
	if (!isfinite(R[0][0]) || !isfinite(R[0][1]) || !isfinite(R[0][2]) ||
		!isfinite(R[1][0]) || !isfinite(R[1][1]) || !isfinite(R[1][2]) ||
		!isfinite(R[2][0]) || !isfinite(R[2][1]) || !isfinite(R[2][2])) {
		pos_pred_out[idx] = pi;
		return;
	}

	// 6. [핵심] 뒤집힘(Reflection) 방어 가드
	// 행렬식이 음수면 덩어리가 뒤집힌 것이므로, 가장 작은 스케일 축(U의 첫 번째 열)을 뒤집음.
	// (오름차순 정렬을 해주므로 무조건 0번 인덱스가 제일 작음)
	if (glm::determinant(R) < 0.0f) {
		U[0] *= -1.0f;
		R = U * glm::transpose(V);
		if (!isfinite(R[0][0]) || !isfinite(R[0][1]) || !isfinite(R[0][2]) ||
			!isfinite(R[1][0]) || !isfinite(R[1][1]) || !isfinite(R[1][2]) ||
			!isfinite(R[2][0]) || !isfinite(R[2][1]) || !isfinite(R[2][2])) {
			pos_pred_out[idx] = pi;
			return;
		}
	}

	}   // ── 기존 float32 경로 끝 ──

	const float3 qi_f3 = f3_sub(pos_rest[idx], comRest);// qi = 자기 자신의 rest 상대 위치 (무게중심 기준)
	glm::vec3 qi(qi_f3.x, qi_f3.y, qi_f3.z);
	//스케일 정보는 버리고 회전 정보 R만 사용. Shape Matching에서 필요한 것은 "이 클러스터가 얼마나 회전했는가"이기 때문
	// R * qi 회전 적용 
	glm::vec3 rotated_qi = R * qi;// R * qi = 회전을 적용한 rest 상대 위치


	//!!!내가 rest 상태에서 무게중심으로부터 qi 방향에 있었는데, 이 클러스터가 R만큼 회전했으니까 나는 지금 comCurr + R*qi 위치에 있어야 해!!!
	//shape matching의 핵심. 변형이 없다면 R=I이고 goal = comCurr + qi가 되어 자연스럽게 원래 형상으로 돌아옴
	const float3 goal = f3_add(comCurr, make_float3(rotated_qi.x, rotated_qi.y, rotated_qi.z));// goal = 현재 무게중심 + 회전된 rest 상대 위치
	const float3 goalDelta = f3_sub(goal, pi);
	const float goalDeltaLen2 = f3_len2(goalDelta);
	// Preserve exact rest-state fixed point against tiny numerical noise.
	if (!isfinite(goalDeltaLen2) || goalDeltaLen2 < 1e-18f) {
		pos_pred_out[idx] = pi;
		return;
	}

	// ... (아래 delta 및 XPBD 당기기 코드는 그대로 유지) ...
	//// Eigen 3x3 행렬 초기화
	//Eigen::Matrix3f A = Eigen::Matrix3f::Zero();
	//
	//{//나 자신의 상대좌표 외적
	//	const float3 q = f3_sub(pos_rest[idx], comRest);
	//	const float3 p = f3_sub(pi, comCurr);
	//	A(0, 0) += p.x * q.x; A(0, 1) += p.x * q.y; A(0, 2) += p.x * q.z;
	//	A(1, 0) += p.y * q.x; A(1, 1) += p.y * q.y; A(1, 2) += p.y * q.z;
	//	A(2, 0) += p.z * q.x; A(2, 1) += p.z * q.y; A(2, 2) += p.z * q.z;
	//}
	////이웃들의 상대좌표 외적 누적
	//for (int k = 0; k < cnt; ++k) {
	//	const int nIdx = nbrIdx[off + k];
	//	if (nIdx < 0 || nIdx >= N) continue;
	//	const float3 q = f3_sub(pos_rest[nIdx], comRest);
	//	const float3 p = f3_sub(pos_pred_in[nIdx], comCurr);
	//	A(0, 0) += p.x * q.x; A(0, 1) += p.x * q.y; A(0, 2) += p.x * q.z;
	//	A(1, 0) += p.y * q.x; A(1, 1) += p.y * q.y; A(1, 2) += p.y * q.z;
	//	A(2, 0) += p.z * q.x; A(2, 1) += p.z * q.y; A(2, 2) += p.z * q.z;
	//}
	//// 회전 행렬 R 추출!
	//Eigen::Matrix3f R;
	//extractRotationIterativeEigen(A, R);
	// 
	//// [해결책 2] NaN 폭발 전염 차단 가드 (안전장치)
	//// 만약 극분해 함수가 버그를 일으켜 R 행렬이 터졌다면, 움직이지 않고 연산 포기!
	//if (isnan(R(0, 0)) || isinf(R(0, 0))) {
	//	pos_pred_out[idx] = pi;
	//	return;
	//}
	//
	//// 5. 목표 위치(Goal) 계산 및 XPBD로 쫀득하게 당기기
	//const float3 qi = f3_sub(pos_rest[idx], comRest);
	//// Eigen 회전 행렬을 float3 벡터(qi)에 곱해주는 변환 과정
	//float3 rotated_qi = make_float3(
	//	R(0, 0) * qi.x + R(0, 1) * qi.y + R(0, 2) * qi.z,
	//	R(1, 0) * qi.x + R(1, 1) * qi.y + R(1, 2) * qi.z,
	//	R(2, 0) * qi.x + R(2, 1) * qi.y + R(2, 2) * qi.z
	//);
	//
	//const float3 goal = f3_add(comCurr, rotated_qi);
	pos_pred_out[idx] = xpbdProjectTowardGoal(//XPBD로 goal 방향으로 당기기
		pi, goal, wi, dt, shapeCompliance, shapeBlend);
}


__global__ void xpbdAngleConstraintKernel(
	int N,
	const float3* pos_pred_in,
	float3* pos_pred_out,
	const float* invMass,
	const int* offset,
	const int* nbrCount,
	const int* nbrIdx,
	const int* pairNextIdx,
	const float* restDot,
	float dt,
	float angleCompliance,
	float angleBlend)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	const float3 p0 = pos_pred_in[idx];
	const float w0 = invMass[idx];
	if (w0 <= 0.0f || pairNextIdx == nullptr || restDot == nullptr) {
		pos_pred_out[idx] = p0;
		return;
	}

	const int off = offset[idx];
	const int cnt = nbrCount[idx];
	if (cnt < 2) {
		pos_pred_out[idx] = p0;
		return;
	}

	const float dt2 = fmaxf(dt * dt, 1e-8f);
	const float alpha = fmaxf(angleCompliance, 0.0f) / dt2;
	float3 accum = make_float3(0.0f, 0.0f, 0.0f);
	int validPairs = 0;
	float avgPairLen = 0.0f;

	for (int k = 0; k < cnt; ++k) {
		const int e0 = off + k;
		const int i1 = nbrIdx[e0];
		const int i2 = pairNextIdx[e0];
		if (i1 < 0 || i1 >= N || i2 < 0 || i2 >= N || i1 == i2) continue;

		const float3 p1 = pos_pred_in[i1];
		const float3 p2 = pos_pred_in[i2];
		const float3 v1 = f3_sub(p1, p0);
		const float3 v2 = f3_sub(p2, p0);
		const float l1 = f3_len(v1);
		const float l2 = f3_len(v2);
		if (l1 < 1e-8f || l2 < 1e-8f) continue;

		// Legacy angle metric: C = v1·v2 - restDot.
		// We keep this behavior for the "original feel", but add strong guards below.
		const float C = f3_dot(v1, v2) - restDot[e0];
		if (!isfinite(C)) continue;

		// Gradients of dot-product constraint.
		const float3 g1 = v2;
		const float3 g2 = v1;
		const float3 g0 = f3_mul(f3_add(v1, v2), -1.0f);

		const float w1 = invMass[i1];
		const float w2 = invMass[i2];
		const float denom =
			w0 * f3_dot(g0, g0) +
			w1 * f3_dot(g1, g1) +
			w2 * f3_dot(g2, g2);
		if (denom < 1e-6f) continue;

		float deltaLambda = -C / (denom + alpha);
		if (!isfinite(deltaLambda)) continue;
		// Safety clamp: prevents sudden spikes from a single malformed pair.
		deltaLambda = fmaxf(fminf(deltaLambda, 0.05f), -0.05f);

		const float3 dp0 = f3_mul(g0, deltaLambda * w0);
		if (!isfinite(dp0.x) || !isfinite(dp0.y) || !isfinite(dp0.z)) continue;
		accum = f3_add(accum, dp0);
		++validPairs;
		avgPairLen += 0.5f * (l1 + l2);
	}

	float3 outPos = p0;
	if (validPairs > 0) {
		const float invCount = 1.0f / float(validPairs);
		float3 avg = f3_mul(accum, invCount);

		// Final step clamp in world units to avoid "self-spinning" runaway.
		avgPairLen *= invCount;
		const float maxStep = fmaxf(1e-5f, 0.1f * avgPairLen);
		const float avgLen = f3_len(avg);
		if (avgLen > maxStep && avgLen > 1e-8f) {
			avg = f3_mul(avg, maxStep / avgLen);
		}
		outPos = f3_add(p0, f3_mul(avg, fminf(fmaxf(angleBlend, 0.0f), 1.0f)));
	}
	pos_pred_out[idx] = outPos;
}

__global__ void xpbdSolveJacobiKernel(//제약조건 솔버 XPBD메인
	int N,
	const float3* pos_pred_in,
	float3* pos_pred_out,
	const float* invMass,
	const int* offset,
	const int* nbrCount,
	const int* nbrIdx,
	const float* nbrDist,
	const float* nbrStiff,
	const float* lambda_in,
	float* lambda_out,
	float dt,
	float invMassScale,
	float stiffnessScale,
	float underRelax,
	float globalCompliance)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	const float3 pi = pos_pred_in[idx];//예측된 좌표 
	const float wi = invMass[idx] * invMassScale;//질량
	float3 accum = make_float3(0.0f, 0.0f, 0.0f);
	int validCount = 0;

	const int off = offset[idx];
	const int cnt = nbrCount[idx];//이웃 수
	const float dt2 = fmaxf(dt * dt, 1e-8f);//프레임이 흘러간 시간의 제곱

	for (int k = 0; k < cnt; ++k) {//이웃 수 만큼 순회
		const int edge = off + k;
		const int j = nbrIdx[edge];
		float lambda = lambda_in[edge];
		if (j < 0 || j >= N) {
			lambda_out[edge] = lambda;
			continue;
		}
		const float3 pj = pos_pred_in[j];
		const float3 diff = f3_sub(pi, pj);
		const float d = f3_len(diff);
		if (d < 1e-7f) {
			lambda_out[edge] = lambda;
			continue;
		}

		const float L0 = nbrDist[edge];//예측전 점들 사이 기존 거리
		const float C = d - L0;//현재 거리d 와 기존거리 L0 의 차이 =  C_j(x_i) 
		const float3 grad = f3_mul(diff, 1.0f / d);// 어느 방향으로 당겨야하는지 정규화된 벡터
		const float wj = invMass[j] * invMassScale;
		const float stiff = fmaxf(nbrStiff[edge] * stiffnessScale, 1e-6f);

		//물질 고유의 유연함(슬라임인지 강철인지)
		const float alpha = globalCompliance / stiff;

		//PBD 와 XPBD 차이 
		//유연함 상수 를 시간의 제곱으로 나누어줌으로 프레임수(컴퓨터 성능)와 솔버 반복횟수와 상관없이 일정한 거동유지.
		const float alpha_tilde = alpha / dt2;

		//두 점이 서로를 당길 때, 질량에 비례해서 얼마나 쉽게 끌려오는지를 나타내는 값
		//분모를 구할 때 기울기 벡터의 크기가 1이라는 점을 이용해(wi + wj)로 최적화
		const float denom = (wi + wj) + alpha_tilde;
		if (denom < 1e-8f) {
			lambda_out[edge] = lambda;
			continue;
		}

		//XPBD equation(18)
		//이번 루프에서 고무줄의 장력(라그랑주 승수, lambda)을 얼마나 변화시켜야 하는가?
		const float delta_lambda = (-C - alpha_tilde * lambda) / denom; 
		lambda += delta_lambda;
		lambda_out[edge] = lambda;

		if (wi > 0.0f) {
			//XPBD equation(17)
			const float3 dp = f3_mul(grad, wi * delta_lambda);//구해진 장력 변화량(Delta lambda)을 이용해, 실제로 내 위치(x_i)를 얼마나 이동시킬 것인가?
			accum = f3_add(accum, dp);
			validCount++;
		}
	}

	float3 outPos = pi;
	if (wi > 0.0f && validCount > 0) {
		const float3 avg = f3_mul(accum, 1.0f / float(validCount));
		outPos = f3_add(pi, f3_mul(avg, underRelax));
	}
	pos_pred_out[idx] = outPos;
}

// 거리 제약 Gauss-Seidel: 한 색(점을 공유하지 않는 간선 묶음)을 스레드당 간선 1개로 푼다.
// Jacobi 경로와 같은 XPBD Eq.18 이지만 (1) 두 끝점을 함께 움직이고 (2) 결과를 바로 pos 에
// 써서 다음 색이 갱신된 위치를 보므로 평균·과소이완이 필요 없다.
__global__ void xpbdDistanceGSColorKernel(
	int begin,
	int count,
	int N,
	float3* pos,
	const float* invMass,
	const int2* edge,
	const float* rest,
	const float* stiff,
	float* lambda,
	float dt,
	float invMassScale,
	float stiffnessScale,
	float globalCompliance)
{
	const int t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= count) return;
	const int e = begin + t;
	const int2 ij = edge[e];
	if (ij.x < 0 || ij.x >= N || ij.y < 0 || ij.y >= N) return;

	const float wi = invMass[ij.x] * invMassScale;
	const float wj = invMass[ij.y] * invMassScale;
	if (wi + wj <= 0.0f) return;

	const float3 pi = pos[ij.x];
	const float3 pj = pos[ij.y];
	const float3 diff = f3_sub(pi, pj);
	const float d = f3_len(diff);
	if (d < 1e-7f) return;

	const float dt2 = fmaxf(dt * dt, 1e-8f);
	const float s = fmaxf(stiff[e] * stiffnessScale, 1e-6f);
	const float alpha_tilde = (globalCompliance / s) / dt2;
	const float denom = (wi + wj) + alpha_tilde;
	if (denom < 1e-8f) return;

	const float C = d - rest[e];
	const float lam = lambda[e];
	const float delta_lambda = (-C - alpha_tilde * lam) / denom;
	lambda[e] = lam + delta_lambda;

	const float3 n = f3_mul(diff, 1.0f / d);
	if (wi > 0.0f) pos[ij.x] = f3_add(pi, f3_mul(n, wi * delta_lambda));
	if (wj > 0.0f) pos[ij.y] = f3_sub(pj, f3_mul(n, wj * delta_lambda));
}

__global__ void xpbdUpdateVelocityKernel(
	int N,
	const float3* pos_curr,
	const float3* pos_pred_final,
	float3* vel,
	float3* pos_curr_out,
	const float* invMass,
	float dt,
	float velDamping)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	const float3 x = pos_curr[idx];//이번 루프 전 원래위치
	const float3 p = pos_pred_final[idx];// 솔버에서 계산한 위치
	const float w = invMass[idx];

	// Stable velocity update path.
	// velDamping is interpreted as damping amount: 0=no damping, 1=full damping.
	if (w > 0.0f) {
		const float3 dx = f3_sub(p, x);
		const float dx2 = f3_len2(dx);
		if (!isfinite(dx2) || dx2 < 1e-18f) {
			vel[idx] = make_float3(0.0f, 0.0f, 0.0f);
			pos_curr_out[idx] = p;
			return;
		}
		float3 v_stable = f3_mul(dx, 1.0f / fmaxf(dt, 1e-8f));
		const float damping = fminf(fmaxf(velDamping, 0.0f), 1.0f);
		const float retention = 1.0f - damping;
		v_stable = f3_mul(v_stable, retention);
		if (!isfinite(v_stable.x) || !isfinite(v_stable.y) || !isfinite(v_stable.z) || f3_len2(v_stable) < 1e-16f) {
			v_stable = make_float3(0.0f, 0.0f, 0.0f);
		}
		vel[idx] = v_stable;
		pos_curr_out[idx] = p;
		return;
	}

	if (w > 0.0f) {
		//XPBD Algorithm 1 Line 16 : update velocities
		float3 v = f3_mul(f3_sub(p, x), 1.0f / fmaxf(dt, 1e-8f));//강제로 위치리를 이동시킨 변화량 을 dt으로 나누어 역으로 속도를 구함. 
		v = f3_mul(v, velDamping);//감쇠, 역산된속도에 마찰력 상수 적용.
		vel[idx] = v;
	}
	else {//invMass가 0인 점(마우스)은 끌려간 것이 아니라 내가 강제로 움직인 것이므로 속도를 0으로 리셋
		vel[idx] = make_float3(0.0f, 0.0f, 0.0f);
	}
	pos_curr_out[idx] = p;
}

// ── Object mode 커널 ───────────────────────────────────────────────────────
// 자기 물체의 강체 속도장 Δv + Δω × (x − c)를 더한다 (물체 단위 반발 충격량).
// imp[10·id] = c(3), Δv(3), Δω(3), fired. 물체가 아닌 입자(id −1)와 마우스로 잡은 점(w=0)은 제외.
__global__ void objectAddRigidVelocityKernel(
	int N,
	const float3* pos,
	float3* vel,
	const float* invMass,
	const int* objId,
	const float* imp)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;
	const int id = objId[idx];
	if (id < 0 || invMass[idx] <= 0.0f) return;
	const float* J = &imp[10 * id];
	if (J[9] == 0.0f) return;
	const float3 p = pos[idx];
	const float rx = p.x - J[0], ry = p.y - J[1], rz = p.z - J[2];
	const float3 v = vel[idx];
	vel[idx] = make_float3(
		v.x + J[3] + (J[7] * rz - J[8] * ry),
		v.y + J[4] + (J[8] * rx - J[6] * rz),
		v.z + J[5] + (J[6] * ry - J[7] * rx));
}

// 모든 입자에 같은 강체 속도장 Δv + Δω × (x − c) 를 더한다 (외부 엔진과의 물체 단위 충돌 충격량).
__global__ void addRigidVelocityAllKernel(int N, const float3* pos, float3* vel, const float* invMass,
	float3 dv, float3 dw, float3 c)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N || invMass[idx] <= 0.0f) return;
	const float3 p = pos[idx];
	const float rx = p.x - c.x, ry = p.y - c.y, rz = p.z - c.z;
	const float3 v = vel[idx];
	vel[idx] = make_float3(v.x + dv.x + (dw.y * rz - dw.z * ry),
		v.y + dv.y + (dw.z * rx - dw.x * rz),
		v.z + dv.z + (dw.x * ry - dw.y * rx));
}

__global__ void attachApplyKernel(int n, const int* idx, const float* pos, int N, float3* posCurr, float3* vel,
	float* invMass, float* objW, float weight)
{
	const int k = blockIdx.x * blockDim.x + threadIdx.x;
	if (k >= n) return;
	const int i = idx[k];
	if (i < 0 || i >= N) return;
	invMass[i] = 0.0f;
	posCurr[i] = make_float3(pos[3 * k], pos[3 * k + 1], pos[3 * k + 2]);
	vel[i] = make_float3(0.0f, 0.0f, 0.0f);
	objW[i] = weight;
}

__global__ void fillFloatKernel(int N, float* a, float v)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) a[i] = v;
}

void FORWARD::setContactShape(bool colliders, bool ground)
{
	g_contactShapeOn = colliders;
	g_contactShapeGround = ground;
}

bool FORWARD::updateContactShapes(const float* scales, const float* quats, const float* opacity, int n, float tau,
	float radiusCap)
{
	if (n <= 0 || !scales || !quats || !opacity) return false;
	if (n != g_contactN) {
		if (d_contactM) cudaFree(d_contactM);
		if (d_contactRmax) cudaFree(d_contactRmax);
		if (d_contactRg) cudaFree(d_contactRg);
		cudaMalloc(&d_contactM, sizeof(float) * 6 * n);
		cudaMalloc(&d_contactRmax, sizeof(float) * n);
		cudaMalloc(&d_contactRg, sizeof(float) * n);
		cudaMemset(d_contactRg, 0, sizeof(float) * n);
		g_contactN = n;
	}
	const int threads = 256;
	contactShapePackKernel << <(n + threads - 1) / threads, threads >> > (n, scales, quats, opacity, tau,
		(isfinite(radiusCap) && radiusCap > 0.0f) ? radiusCap : 0.0f, d_contactM, d_contactRmax);
	return cudaGetLastError() == cudaSuccess;
}

void FORWARD::setAttachedParticles(int count, const int* idx, const float* pos, float weight)
{
	if (!d_invMass || !d_pos_curr || !d_vel || cm_num_elements <= 0) return;
	const int N = cm_num_elements;
	if (g_attActive) cudaMemcpy(d_invMass, d_invMassSaved, sizeof(float) * N, cudaMemcpyDeviceToDevice);   // 지난번 것을 푼다
	if (count <= 0 || !idx || !pos) {
		g_attActive = false;
		g_attCount = 0;
		return;
	}
	if (!g_attActive) {
		if (!d_invMassSaved) cudaMalloc(&d_invMassSaved, sizeof(float) * N);
		if (!d_objW) cudaMalloc(&d_objW, sizeof(float) * N);
		cudaMemcpy(d_invMassSaved, d_invMass, sizeof(float) * N, cudaMemcpyDeviceToDevice);
	}
	fillFloatKernel << <(N + 255) / 256, 256 >> > (N, d_objW, 1.0f);
	if (count > g_attCap) {
		if (d_attIdx) cudaFree(d_attIdx);
		if (d_attPos) cudaFree(d_attPos);
		cudaMalloc(&d_attIdx, sizeof(int) * count);
		cudaMalloc(&d_attPos, sizeof(float) * 3 * count);
		g_attCap = count;
	}
	cudaMemcpy(d_attIdx, idx, sizeof(int) * count, cudaMemcpyHostToDevice);
	cudaMemcpy(d_attPos, pos, sizeof(float) * 3 * count, cudaMemcpyHostToDevice);
	attachApplyKernel << <(count + 255) / 256, 256 >> > (count, d_attIdx, d_attPos, N, d_pos_curr, d_vel, d_invMass, d_objW,
		fmaxf(weight, 1.0f));
	g_attActive = true;
	g_attCount = count;
}

void FORWARD::addRigidVelocity(const float dv[3], const float dw[3], const float center[3])
{
	if (!d_vel || !d_pos_curr || !d_invMass || cm_num_elements <= 0) return;
	const int N = cm_num_elements;
	addRigidVelocityAllKernel << <(N + 255) / 256, 256 >> > (N, d_pos_curr, d_vel, d_invMass,
		make_float3(dv[0], dv[1], dv[2]), make_float3(dw[0], dw[1], dw[2]), make_float3(center[0], center[1], center[2]));
}

// 물체별 shape matching (Müller et al. 2005): goal = c + R (x_rest − c0),  p += s·(goal − p).
// params[16·id] = c0(3), c(3), R 행우선(9), valid. c·R은 호스트가 물체마다 double로 누적한
// A = Σ (p − c)(x_rest − c0)ᵀ 의 극분해 (회전·이동은 자유). 물체가 아닌 입자(id −1)는 건드리지 않는다.
__global__ void objectShapeMatchKernel(
	int N,
	float3* pos,
	const float3* pos_rest,
	const float* invMass,
	const int* objId,
	const float* params,
	float s)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;
	const int id = objId[idx];
	if (id < 0 || invMass[idx] <= 0.0f) return;
	const float* P = &params[16 * id];
	if (P[15] == 0.0f) return;
	const float3 r = pos_rest[idx];
	const float qx = r.x - P[0], qy = r.y - P[1], qz = r.z - P[2];
	const float3 p = pos[idx];
	const float gx = P[3] + P[6] * qx + P[7] * qy + P[8] * qz;
	const float gy = P[4] + P[9] * qx + P[10] * qy + P[11] * qz;
	const float gz = P[5] + P[12] * qx + P[13] * qy + P[14] * qz;
	pos[idx] = make_float3(p.x + s * (gx - p.x), p.y + s * (gy - p.y), p.z + s * (gz - p.z));
}

// GPU 경로 1/3: 물체별 Σ(p−s) 와 Σ(p−s)qᵀ (q = x_rest − c0) 를 double 로 누적한다.
// s = 직전 프레임 무게중심이라 p−s 가 작다 → 한 번만 읽어도 호스트의 두 번 읽기(무게중심 → 편차)와 같은 정밀도.
// 블록 안에서 먼저 모으고(워프 전체가 한 물체면 shuffle, 아니면 공유 메모리 atomic) 블록당 물체마다 12번만 전역 atomic.
// weight 가 있으면 (붙잡기 중) 항마다 무게를 곱하고 Σw, Σw·q 를 더해 16 개. 없으면 예전과 같은 12 개·같은 산술.
__global__ void objectShapeAccumKernel(
	int N,
	const float3* pos,
	const int* objId,
	const double* restRel,
	const double* shift,
	int K,
	double* sums,
	const float* weight,
	int stride)
{
	__shared__ double sh[OBJ_GPU_MAX_K * 16];
	for (int t = threadIdx.x; t < K * stride; t += blockDim.x) sh[t] = 0.0;
	__syncthreads();

	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int id = (idx < N) ? objId[idx] : -1;
	double v[16];
	if (id >= 0) {
		const float3 p = pos[idx];
		const double dx = (double)p.x - shift[3 * id], dy = (double)p.y - shift[3 * id + 1], dz = (double)p.z - shift[3 * id + 2];
		const double qx = restRel[3 * (size_t)idx], qy = restRel[3 * (size_t)idx + 1], qz = restRel[3 * (size_t)idx + 2];
		v[0] = dx; v[1] = dy; v[2] = dz;
		v[3] = dx * qx; v[4] = dx * qy; v[5] = dx * qz;
		v[6] = dy * qx; v[7] = dy * qy; v[8] = dy * qz;
		v[9] = dz * qx; v[10] = dz * qy; v[11] = dz * qz;
		if (weight) {
			const double w = (double)weight[idx];
			for (int k = 0; k < 12; ++k) v[k] *= w;
			v[12] = w; v[13] = w * qx; v[14] = w * qy; v[15] = w * qz;
		}
	}
	else {
		for (int k = 0; k < 16; ++k) v[k] = 0.0;
	}
	const unsigned int full = 0xffffffffu;
	const int id0 = __shfl_sync(full, id, 0);
	if (__all_sync(full, id == id0)) {
		if (id0 >= 0) {
			for (int k = 0; k < stride; ++k)
				for (int o = 16; o > 0; o >>= 1) v[k] += __shfl_xor_sync(full, v[k], o);
			if ((threadIdx.x & 31) == 0)
				for (int k = 0; k < stride; ++k) atomicAdd(&sh[id0 * stride + k], v[k]);
		}
	}
	else if (id >= 0) {
		for (int k = 0; k < stride; ++k) atomicAdd(&sh[id * stride + k], v[k]);
	}
	__syncthreads();
	for (int t = threadIdx.x; t < K * stride; t += blockDim.x)
		if (sh[t] != 0.0) atomicAdd(&sums[t], sh[t]);
}

// GPU 경로 2/3: 물체당 1스레드. c = s + Σ(p−s)/n,  A = Σ(p−s)qᵀ − (c−s)(Σq)ᵀ = Σ(p−c)qᵀ,
// 극분해는 형상 제약과 같은 double 경로(shapeMatchRotationDouble, Frobenius 정규화 입력).
// params 형식은 호스트 경로와 같다. 다음 프레임 기준 s 를 c 로 바꾸고 이동량 |c − s| 를 남긴다.
// stride 16 (붙잡기 중) 이면 개수·Σq 대신 누적한 Σw·Σw·q 를 쓰고, rest 쪽 기준점도 무게 중심 c0 + Σw·q/Σw 로 옮긴다
// (옮기지 않으면 goal 이 R·(Σw·q/Σw) 만큼 어긋난다).
__global__ void objectShapeSolveKernel(
	int K,
	const double* sums,
	int stride,
	const int* count,
	const double* sumQ,
	const float* c0,
	double* shift,
	float* params,
	float* disp)
{
	const int k = blockIdx.x * blockDim.x + threadIdx.x;
	if (k >= K) return;
	float* P = &params[16 * k];
	const int n = count[k];
	if (n < 4) { P[15] = 0.0f; disp[k] = 0.0f; return; }
	const double* S = &sums[stride * k];
	const bool wOn = (stride == 16);
	const double W = wOn ? S[12] : (double)n;
	if (!(W > 0.0)) { P[15] = 0.0f; disp[k] = 0.0f; return; }
	const double inv = 1.0 / W;
	const double dc[3] = { S[0] * inv, S[1] * inv, S[2] * inv };
	const double* sq = wOn ? &S[13] : &sumQ[3 * k];
	double a[9];
	for (int r = 0; r < 3; ++r)
		for (int c = 0; c < 3; ++c)
			a[3 * r + c] = S[3 + 3 * r + c] - dc[r] * sq[c];
	double cNow[3];
	for (int t = 0; t < 3; ++t) {
		cNow[t] = shift[3 * k + t] + dc[t];
		shift[3 * k + t] = cNow[t];
	}
	disp[k] = (float)sqrt(dc[0] * dc[0] + dc[1] * dc[1] + dc[2] * dc[2]);

	double fn = 0.0;
	for (int t = 0; t < 9; ++t) fn += a[t] * a[t];
	fn = sqrt(fn);
	if (!(fn > 1e-300) || !isfinite(fn)) { P[15] = 0.0f; return; }
	glm::mat3 A;
	for (int r = 0; r < 3; ++r)
		for (int c = 0; c < 3; ++c)
			A[c][r] = (float)(a[3 * r + c] / fn);
	glm::mat3 R;
	if (!shapeMatchRotationDouble(A, R)) { P[15] = 0.0f; return; }
	for (int t = 0; t < 3; ++t) {
		P[t] = wOn ? (float)((double)c0[3 * k + t] + S[13 + t] * inv) : c0[3 * k + t];
		P[3 + t] = (float)cNow[t];
		for (int cc = 0; cc < 3; ++cc) P[6 + 3 * t + cc] = R[cc][t];   // 행우선 R(t, cc)
	}
	P[15] = 1.0f;
}

// GPU 경로 3/3 (자기충돌이 켜졌을 때만): 물체별 경계상자(predict 위치). float 비트를 정수로 비교하는 atomic min/max.
__device__ __forceinline__ void objAtomicMinF(float* a, float v)
{
	v += 0.0f;   // −0 → +0
	if (v >= 0.0f) atomicMin((int*)a, __float_as_int(v));
	else atomicMax((unsigned int*)a, __float_as_uint(v));
}
__device__ __forceinline__ void objAtomicMaxF(float* a, float v)
{
	v += 0.0f;
	if (v >= 0.0f) atomicMax((int*)a, __float_as_int(v));
	else atomicMin((unsigned int*)a, __float_as_uint(v));
}

__global__ void objectBoxesInitKernel(int K, float* boxes)
{
	const int t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t < 6 * K) boxes[t] = ((t % 6) < 3) ? FLT_MAX : -FLT_MAX;
}

__global__ void objectBoxesKernel(int N, const float3* pos, const int* objId, int K, float* boxes)
{
	__shared__ float sh[OBJ_GPU_MAX_K * 6];
	for (int t = threadIdx.x; t < K * 6; t += blockDim.x) sh[t] = ((t % 6) < 3) ? FLT_MAX : -FLT_MAX;
	__syncthreads();
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N) {
		const int id = objId[idx];
		const float3 p = pos[idx];
		if (id >= 0 && isfinite(p.x) && isfinite(p.y) && isfinite(p.z)) {
			float* b = &sh[6 * id];
			objAtomicMinF(&b[0], p.x); objAtomicMinF(&b[1], p.y); objAtomicMinF(&b[2], p.z);
			objAtomicMaxF(&b[3], p.x); objAtomicMaxF(&b[4], p.y); objAtomicMaxF(&b[5], p.z);
		}
	}
	__syncthreads();
	for (int t = threadIdx.x; t < K * 6; t += blockDim.x) {
		const float v = sh[t];
		if (v == FLT_MAX || v == -FLT_MAX) continue;
		if ((t % 6) < 3) objAtomicMinF(&boxes[t], v);
		else objAtomicMaxF(&boxes[t], v);
	}
}

// ── Self-collision 커널 ─────────────────────────────────────────────────────
__device__ unsigned int g_selfColCtr[3];   // [0] 후보 수 합, [1] 상한 초과로 교체한 횟수, [2] 활성 입자 수

__device__ __forceinline__ unsigned int selfColHash(int cx, int cy, int cz, unsigned int mask)
{
	return (((unsigned int)cx * 73856093u) ^ ((unsigned int)cy * 19349663u) ^ ((unsigned int)cz * 83492791u)) & mask;
}

__device__ __forceinline__ int selfColCell(float v, float invCell)
{
	return (int)floorf(fminf(fmaxf(v * invCell, -1.0e9f), 1.0e9f));   // 거대 좌표의 int 변환 UB 방지
}

// 다른 물체 경계상자(부풀림 포함) 안에 predict 위치나 시작 위치가 들어간 입자만 활성.
// 물체가 아닌 입자(floater, id −1)는 크기가 작아 항상 활성으로 둔다.
__global__ void selfColActiveKernel(
	int N,
	const float3* posStart,
	const float3* posPred,
	const int* objId,
	const float* boxes,
	int K,
	float inflate,
	unsigned char* active)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	const int id = objId[i];
	unsigned char a = 0;
	if (id < 0) {
		a = 1;
	}
	else {
		const float3 p = posPred[i];
		const float3 x = posStart[i];
		for (int k = 0; k < K; ++k) {
			if (k == id) continue;
			const float* b = &boxes[6 * k];
			const float lx = b[0] - inflate, ly = b[1] - inflate, lz = b[2] - inflate;
			const float hx = b[3] + inflate, hy = b[4] + inflate, hz = b[5] + inflate;
			const bool inP = p.x >= lx && p.x <= hx && p.y >= ly && p.y <= hy && p.z >= lz && p.z <= hz;
			const bool inX = x.x >= lx && x.x <= hx && x.y >= ly && x.y <= hy && x.z >= lz && x.z <= hz;
			if (inP || inX) { a = 1; break; }
		}
	}
	active[i] = a;
	if (a) atomicAdd(&g_selfColCtr[2], 1u);
}

// 격자 칸 키. 해시 충돌은 거리 검사로 걸러지므로 정확성에는 영향이 없다.
// 비활성 입자는 sentinel(= 테이블 크기) 키를 받아 정렬 끝으로 가고 어떤 칸에도 들어가지 않는다.
__global__ void selfColKeyKernel(int N, const float3* pos, const unsigned char* active, unsigned int* keys, int* idx,
	float invCell, unsigned int mask, unsigned int sentinel)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	const float3 p = pos[i];
	idx[i] = i;
	if (active != nullptr && !active[i]) { keys[i] = sentinel; return; }
	if (!isfinite(p.x) || !isfinite(p.y) || !isfinite(p.z)) { keys[i] = sentinel; return; }
	keys[i] = selfColHash(selfColCell(p.x, invCell), selfColCell(p.y, invCell), selfColCell(p.z, invCell), mask);
}

// 정렬된 키에서 칸별 [start, end) 구간. cellStart는 호출 전에 -1로 채운다. sentinel 키는 건너뛴다.
__global__ void selfColCellRangeKernel(int N, const unsigned int* keysSorted, int* cellStart, int* cellEnd, unsigned int mask)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	const unsigned int k = keysSorted[i];
	if (k > mask) return;
	if (i == 0 || keysSorted[i - 1] != k) cellStart[k] = i;
	if (i == N - 1 || keysSorted[i + 1] != k) cellEnd[k] = i + 1;
}

// 이번 프레임 predict 변위² (최대값을 구해 격자 칸 크기를 정한다)
__global__ void selfColDispKernel(int N, const float3* posStart, const float3* posPred, const unsigned char* active, float* out)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	if (active != nullptr && !active[i]) { out[i] = 0.0f; return; }   // 칸 크기는 활성 입자의 이동량으로만 정한다
	const float3 x = posStart[i];
	const float3 p = posPred[i];
	const float dx = p.x - x.x, dy = p.y - x.y, dz = p.z - x.z;
	const float d2 = dx * dx + dy * dy + dz * dz;
	out[i] = isfinite(d2) ? d2 : 0.0f;
}

// 입자별 후보 목록. 격자는 '프레임 시작 위치' x 로 만든다 — predict 로 이미 한 프레임 이동한 위치에서 찾으면
// 빠른 물체가 얇은 막을 건너뛴 뒤라 후보가 아예 안 잡힌다
// (CPU 복제 tunnel_experiment2.py: 고정 접시에 4.2·d_c/frame 로 떨어진 쿠션 100% 관통, 후보 0개).
//   다른 물체 쌍: |x_i − x_j| < search + |d_i − d_j|  (d = p − x, 이번 프레임에 가까워질 수 있는 최대량)
//                rest 에서 이미 겹쳐 있던 쌍만 제외. 목록에서 항상 우선 — 충돌 순간 같은 물체 안의 상대 이동이 커져
//                같은 물체 후보가 32칸을 채우고 정작 다른 물체 후보를 밀어내던 것을 막는다.
//                상대 이동 > fastRel 이면 '빠른 접근' 쌍 → 인덱스를 −(j+1) 로 저장 (해결 커널이 평균 법선 규칙 사용)
//   같은 물체 쌍: predict 위치에서 |p_i − p_j| < search, 처음부터 붙어 있던 쌍 제외 (기존 규칙)
// 목록이 차면 (같은 물체 쌍 먼저, 같은 등급이면 늦게 닿을 쌍부터) 교체한다. 같은 해시 칸은 한 번만 본다.
__global__ void selfColBuildKernel(
	int N,
	const float3* posStart,
	const float3* posPred,
	const float3* rest,
	const int* idxSorted,
	const int* cellStart,
	const int* cellEnd,
	float invCell,
	unsigned int mask,
	float search,
	float excludeR2,
	const int* objId,          // nullptr 이면 전부 같은 물체로 본다
	float crossExcludeR2,      // 서로 다른 물체 쌍의 제외 거리² (rest 에서 이미 겹쳐 있던 쌍만 제외)
	float fastRel,             // 다른 물체 쌍의 상대 이동이 이보다 크면 평균 법선 규칙
	const unsigned char* active,   // nullptr 이면 전부 활성
	int withinBody,            // 0 이면 같은 물체 쌍은 후보로 넣지 않는다
	int* contacts,
	int* counts)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	const float3 x = posStart[i];
	const float3 p = posPred[i];
	int cnt = 0;
	const bool isActive = (active == nullptr) || (active[i] != 0);
	if (isActive && isfinite(x.x) && isfinite(x.y) && isfinite(x.z) && isfinite(p.x) && isfinite(p.y) && isfinite(p.z)) {
		const float3 q = rest[i];
		const float dix = p.x - x.x, diy = p.y - x.y, diz = p.z - x.z;
		const int oi = (objId != nullptr) ? objId[i] : 0;
		const float search2 = search * search;
		const int cx = selfColCell(x.x, invCell), cy = selfColCell(x.y, invCell), cz = selfColCell(x.z, invCell);
		const int base = i * SELFCOL_MAX_CONTACTS;
		unsigned int hs[27];
		float prio[SELFCOL_MAX_CONTACTS];
		int tier[SELFCOL_MAX_CONTACTS];    // 0 = 다른 물체, 1 = 같은 물체
		int nh = 0;
		for (int oz = -1; oz <= 1; ++oz)
		for (int oy = -1; oy <= 1; ++oy)
		for (int ox = -1; ox <= 1; ++ox) {
			const unsigned int h = selfColHash(cx + ox, cy + oy, cz + oz, mask);
			bool dup = false;
			for (int m = 0; m < nh; ++m) { if (hs[m] == h) { dup = true; break; } }
			if (dup) continue;
			hs[nh++] = h;
			const int s = cellStart[h];
			if (s < 0) continue;
			const int e = cellEnd[h];
			for (int t = s; t < e; ++t) {
				const int j = idxSorted[t];
				if (j == i) continue;
				const bool cross = (objId != nullptr) && (objId[j] != oi);
				if (!cross && !withinBody) continue;
				const float3 pj = posPred[j];
				const float3 qj = rest[j];
				const float qx = q.x - qj.x, qy = q.y - qj.y, qz = q.z - qj.z;
				const float q2 = qx * qx + qy * qy + qz * qz;
				float pr;
				int code = j;
				if (cross) {
					const float3 xj = posStart[j];
					const float ex = x.x - xj.x, ey = x.y - xj.y, ez = x.z - xj.z;
					const float distX = sqrtf(ex * ex + ey * ey + ez * ez);
					const float rx = dix - (pj.x - xj.x), ry = diy - (pj.y - xj.y), rz = diz - (pj.z - xj.z);
					const float rel = sqrtf(rx * rx + ry * ry + rz * rz);
					if (!(distX < search + rel)) continue;    // NaN도 여기서 걸러진다
					if (!(q2 >= crossExcludeR2)) continue;    // rest 에서 이미 겹쳐 있던 쌍
					pr = distX - rel;                         // 작을수록 먼저 닿는다
					if (rel > fastRel) code = -(j + 1);
				}
				else {
					const float ex = p.x - pj.x, ey = p.y - pj.y, ez = p.z - pj.z;
					const float d2 = ex * ex + ey * ey + ez * ez;
					if (!(d2 < search2)) continue;
					if (!(q2 >= excludeR2)) continue;         // 처음부터 붙어 있던 쌍 (그래프가 붙잡고 있다)
					pr = sqrtf(d2);
				}
				const int tr = cross ? 0 : 1;
				if (cnt < SELFCOL_MAX_CONTACTS) {
					contacts[base + cnt] = code;
					prio[cnt] = pr;
					tier[cnt] = tr;
					++cnt;
				}
				else {
					int worst = 0;   // 'far'는 windef.h 매크로라 쓰면 안 된다
					for (int m = 1; m < SELFCOL_MAX_CONTACTS; ++m) {
						if (tier[m] > tier[worst] || (tier[m] == tier[worst] && prio[m] > prio[worst])) worst = m;
					}
					if (tr < tier[worst] || (tr == tier[worst] && pr < prio[worst])) {
						contacts[base + worst] = code;
						prio[worst] = pr;
						tier[worst] = tr;
					}
					atomicAdd(&g_selfColCtr[1], 1u);
				}
			}
		}
	}
	counts[i] = cnt;
	if (cnt > 0) atomicAdd(&g_selfColCtr[0], (unsigned int)cnt);
}

// 빠른 접근 쌍의 기준 방향: 프레임 시작 위치 차이. 시작 위치가 겹쳤으면 rest 차이 방향.
__device__ __forceinline__ bool selfColStartNormal(const float3& xi, const float3& xj, const float3& qi, const float3& qj,
	float& nx, float& ny, float& nz)
{
	nx = xi.x - xj.x; ny = xi.y - xj.y; nz = xi.z - xj.z;
	float l2 = nx * nx + ny * ny + nz * nz;
	if (!(l2 > 1e-24f)) {
		nx = qi.x - qj.x; ny = qi.y - qj.y; nz = qi.z - qj.z;
		l2 = nx * nx + ny * ny + nz * nz;
		if (!(l2 > 1e-24f)) return false;
	}
	const float inv = 1.0f / sqrtf(l2);
	nx *= inv; ny *= inv; nz *= inv;
	return true;
}

// 반복마다 1단계(읽기).
//   일반 쌍(같은 물체 등): d < d_c 이면 겹침을 역질량 비율로 — 입자별 활성 접촉 평균 (같은 질량이면 절반씩)
//   빠른 접근 쌍(인덱스 음수): 시작 방향 n0 의 반공간 (p_i − p_j)·n0 < d_c 가 깨진 쌍들의 n0 를 침투량으로
//     가중 합산한 '한 방향' n̄ 으로만, 모든 그 쌍이 n̄ 방향으로 d_c 이상 떨어지도록 최대 침투만큼 민다.
//     쌍마다 제 방향으로 밀면 옆 성분이 얇은 막을 옆으로 벌려 위 물체가 빠져 내려간다
//     (CPU 복제, 바닥 위 접시: 거리 규칙은 충돌 직후 층 간격 0.01·d_c). 평균 법선은 대칭 이웃의 옆 성분이 상쇄된다
//     (고정 접시 관통 0%·층 간격 1.0·d_c 유지, 바닥 위 접시 89프레임 0.65·d_c).
__global__ void selfColSolveKernel(
	int N,
	const float3* pos,
	const float3* posStart,
	const float3* rest,
	const float* invMass,
	const int* contacts,
	const int* counts,
	float dc,
	float invMassScale,
	float3* dpOut,
	float3* dpCrossOut)        // 위 보정 중 빠른 접근(다른 물체) 성분만 — 접촉 평균 강체 이동의 입력
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	const int cnt = counts[i];
	if (cnt <= 0) return;
	const float wi = invMass[i] * invMassScale;
	float ax = 0.0f, ay = 0.0f, az = 0.0f;
	int act = 0;
	float cpx = 0.0f, cpy = 0.0f, cpz = 0.0f;
	float sx = 0.0f, sy = 0.0f, sz = 0.0f, wFast = 0.0f;
	int nFast = 0;
	if (wi > 0.0f) {
		const float3 pi = pos[i];
		const float3 xi = posStart[i];
		const float3 qi = rest[i];
		const float dc2 = dc * dc;
		const int base = i * SELFCOL_MAX_CONTACTS;
		for (int k = 0; k < cnt; ++k) {
			const int code = contacts[base + k];
			if (code < 0) {
				const int j = -code - 1;
				float nx, ny, nz;
				if (!selfColStartNormal(xi, posStart[j], qi, rest[j], nx, ny, nz)) continue;
				const float3 pj = pos[j];
				const float proj = (pi.x - pj.x) * nx + (pi.y - pj.y) * ny + (pi.z - pj.z) * nz;
				if (!(proj < dc)) continue;
				const float pen = dc - proj;
				sx += nx * pen; sy += ny * pen; sz += nz * pen;
				const float wj = invMass[j] * invMassScale;
				wFast += wi / (wi + wj);
				++nFast;
				continue;
			}
			const int j = code;
			const float3 pj = pos[j];
			const float dx = pi.x - pj.x, dy = pi.y - pj.y, dz = pi.z - pj.z;
			const float d2 = dx * dx + dy * dy + dz * dz;
			if (!(d2 < dc2)) continue;
			const float wj = invMass[j] * invMassScale;
			const float wr = wi / (wi + wj);
			if (d2 < 1e-24f) {
				// 정확히 겹친 쌍은 현재 위치로 방향을 정할 수 없어 예전에는 영원히 붙어 있었다 (외부 검토 지적).
				// 후보는 rest 거리 ≥ d_ex > 0 으로 골랐으므로 rest 차이 방향은 항상 정의되고, i·j가 서로 반대로 민다.
				const float3 qi = rest[i], qj = rest[j];
				const float rx = qi.x - qj.x, ry = qi.y - qj.y, rz = qi.z - qj.z;
				const float r2 = rx * rx + ry * ry + rz * rz;
				if (!(r2 > 1e-24f)) continue;
				const float s = dc * wr / sqrtf(r2);
				ax += rx * s; ay += ry * s; az += rz * s;
				++act;
				continue;
			}
			const float d = sqrtf(d2);
			const float s = (dc - d) / d * wr;
			ax += dx * s; ay += dy * s; az += dz * s;
			++act;
		}
	}
	float3 out = (act > 0) ? make_float3(ax / act, ay / act, az / act) : make_float3(0.0f, 0.0f, 0.0f);
	if (nFast > 0) {
		const float sl = sqrtf(sx * sx + sy * sy + sz * sz);
		if (sl > 1e-20f) {
			const float bx = sx / sl, by = sy / sl, bz = sz / sl;
			// 같은 활성 집합에 대해 n̄ 방향 침투의 최댓값 (충돌 프레임에만 도는 두 번째 순회)
			const float3 pi = pos[i];
			const float3 xi = posStart[i];
			const float3 qi = rest[i];
			const int base = i * SELFCOL_MAX_CONTACTS;
			float delta = 0.0f;
			for (int k = 0; k < cnt; ++k) {
				const int code = contacts[base + k];
				if (code >= 0) continue;
				const int j = -code - 1;
				float nx, ny, nz;
				if (!selfColStartNormal(xi, posStart[j], qi, rest[j], nx, ny, nz)) continue;
				const float3 pj = pos[j];
				const float ddx = pi.x - pj.x, ddy = pi.y - pj.y, ddz = pi.z - pj.z;
				if (!(ddx * nx + ddy * ny + ddz * nz < dc)) continue;
				const float pen = dc - (ddx * bx + ddy * by + ddz * bz);
				if (pen > delta) delta = pen;
			}
			if (delta > 0.0f) {
				const float sc = delta * (wFast / (float)nFast);
				cpx = bx * sc; cpy = by * sc; cpz = bz * sc;
				out = make_float3(out.x + cpx, out.y + cpy, out.z + cpz);
			}
		}
	}
	dpOut[i] = out;
	dpCrossOut[i] = make_float3(cpx, cpy, cpz);
}

// 2단계(쓰기): 1단계가 전부 읽은 뒤 적용하므로 결과가 스레드 실행 순서와 무관하다.
// accum 에는 다른 물체 보정 성분을 프레임 동안 누적한다 (접촉 평균 강체 이동이 읽는다).
__global__ void selfColApplyKernel(int N, float3* pos, const int* counts, const float3* dp, const float3* dpCross, float3* accum)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	if (counts[i] <= 0) return;
	const float3 p = pos[i];
	const float3 d = dp[i];
	pos[i] = make_float3(p.x + d.x, p.y + d.y, p.z + d.z);
	const float3 c = dpCross[i];
	const float3 a = accum[i];
	accum[i] = make_float3(a.x + c.x, a.y + c.y, a.z + c.z);
}

// ── 접촉 평균 강체 이동 커널 ────────────────────────────────────────────────
// 물체별로 '이번 프레임 다른 물체 보정을 받은 점'의 보정 합과 개수를 모은다 (보정 받은 점만 쓰기 — 경합이 적다).
__global__ void objectPushSumKernel(int N, const int* objId, const float3* accum, float* sums)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	const int id = objId[i];
	if (id < 0) return;
	const float3 a = accum[i];
	if (!(a.x * a.x + a.y * a.y + a.z * a.z > 1e-30f)) return;
	atomicAdd(&sums[4 * id], a.x);
	atomicAdd(&sums[4 * id + 1], a.y);
	atomicAdd(&sums[4 * id + 2], a.z);
	atomicAdd(&sums[4 * id + 3], 1.0f);
}

// 보정을 받지 않은 같은 물체 점들에 물체별 평균 이동량을 더한다 (보정 받은 점은 이미 제 몫만큼 움직였다).
__global__ void objectPushApplyKernel(int N, const int* objId, const float* invMass, const float3* accum, const float* pushT, float3* pos)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	const int id = objId[i];
	if (id < 0 || invMass[i] <= 0.0f) return;
	const float* T = &pushT[4 * id];
	if (T[3] == 0.0f) return;
	const float3 a = accum[i];
	if (a.x * a.x + a.y * a.y + a.z * a.z > 1e-30f) return;
	const float3 p = pos[i];
	pos[i] = make_float3(p.x + T[0], p.y + T[1], p.z + T[2]);
}

// ── 타원체 접촉: 모양 묶기 ──────────────────────────────────────────────────
// scales[N×3] (activated), quats[N×4] (w,x,y,z), opacity[N] (activated) → M = k²Σ, rmax = k·s_max.
// rcap > 0 이면 반축마다 rcap 으로 자른다 (튀는 큰 가우시안이 두꺼운 껍질을 만들지 않게).
__global__ void contactShapePackKernel(int N, const float* sc, const float* qt, const float* op, float tau, float rcap,
	float* M, float* rmax)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	const float a = op[i];
	const float k = (a > tau && tau > 0.0f) ? sqrtf(2.0f * logf(a / tau)) : 0.0f;
	float e0 = k * sc[3 * i], e1 = k * sc[3 * i + 1], e2 = k * sc[3 * i + 2];
	if (rcap > 0.0f) { e0 = fminf(e0, rcap); e1 = fminf(e1, rcap); e2 = fminf(e2, rcap); }
	if (!(isfinite(e0) && isfinite(e1) && isfinite(e2))) { e0 = e1 = e2 = 0.0f; }
	float w = qt[4 * i], x = qt[4 * i + 1], y = qt[4 * i + 2], z = qt[4 * i + 3];
	const float qn = sqrtf(w * w + x * x + y * y + z * z);
	if (qn > 1e-20f) { w /= qn; x /= qn; y /= qn; z /= qn; } else { w = 1.0f; x = y = z = 0.0f; }
	// 회전 행렬 (열 = 가우시안 로컬 축)
	const float r00 = 1 - 2 * (y * y + z * z), r01 = 2 * (x * y - w * z), r02 = 2 * (x * z + w * y);
	const float r10 = 2 * (x * y + w * z), r11 = 1 - 2 * (x * x + z * z), r12 = 2 * (y * z - w * x);
	const float r20 = 2 * (x * z - w * y), r21 = 2 * (y * z + w * x), r22 = 1 - 2 * (x * x + y * y);
	const float l0 = e0 * e0, l1 = e1 * e1, l2 = e2 * e2;
	float* m = &M[6 * i];
	m[0] = r00 * r00 * l0 + r01 * r01 * l1 + r02 * r02 * l2;
	m[1] = r00 * r10 * l0 + r01 * r11 * l1 + r02 * r12 * l2;
	m[2] = r00 * r20 * l0 + r01 * r21 * l1 + r02 * r22 * l2;
	m[3] = r10 * r10 * l0 + r11 * r11 * l1 + r12 * r12 * l2;
	m[4] = r10 * r20 * l0 + r11 * r21 * l1 + r12 * r22 * l2;
	m[5] = r20 * r20 * l0 + r21 * r21 * l1 + r22 * r22 * l2;
	rmax[i] = fmaxf(e0, fmaxf(e1, e2));
}

// 법선 n 방향 지지 거리 √(nᵀ M n)
__device__ __forceinline__ float contactSupport(const float* m, float nx, float ny, float nz)
{
	const float q = m[0] * nx * nx + m[3] * ny * ny + m[5] * nz * nz
		+ 2.0f * (m[1] * nx * ny + m[2] * nx * nz + m[4] * ny * nz);
	return sqrtf(fmaxf(q, 0.0f));
}

__global__ void contactGroundRadiusKernel(int N, const float* M, float3 n, float* rg)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	rg[i] = contactSupport(&M[6 * i], n.x, n.y, n.z);
}

// ── Ground contact 커널 ────────────────────────────────────────────────────
// 위치 단계: 평면 아래로 들어간 중심을 법선 방향으로 밀어올린다. 정적 평면이라 질량 가중이 없고
// 입자끼리 독립이라 in-place가 안전하다. 쓰기는 접촉한 입자에서만 일어난다.
// 마우스로 잡은 점(w=0)은 사용자가 위치를 정하므로 건드리지 않는다.
__global__ void groundProjectKernel(
	int N,
	float3* pos,
	const float* invMass,
	float3 n,
	float planeD,   // h + r
	const float* rg)   // 타원체 접촉: 입자별 법선 방향 반경 (nullptr = 점)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;
	if (invMass[idx] <= 0.0f) return;
	const float3 p = pos[idx];
	const float c = n.x * p.x + n.y * p.y + n.z * p.z - planeD - (rg ? rg[idx] : 0.0f);
	if (c < 0.0f) {
		pos[idx] = make_float3(p.x - c * n.x, p.y - c * n.y, p.z - c * n.z);
	}
}

// 운동학 충돌체 투영: 입자가 충돌체 표면 + margin 안에 있으면 가장 가까운 바깥 방향으로 밀어낸다.
// 부호 거리는 충돌체 로컬 좌표에서 계산 (box 는 표준 SDF — 밖이면 가장 가까운 점, 안이면 가장 가까운 면).
__global__ void kinColliderProjectKernel(
	int N,
	float3* pos,
	const float3* posPrev,   // 스텝 시작 위치 x_t (마찰용)
	const float* invMass,
	const KinCollider* cols,
	int nc,
	float margin,
	float mu,
	int* hits,
	float* push,
	int countHits,
	const float* cM,      // 타원체 접촉: k²Σ [N×6] (nullptr = 점)
	const float* cRmax)   //              가장 긴 반축 [N]
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;
	if (invMass[idx] <= 0.0f) return;
	float3 p = pos[idx];
	bool moved = false;
	const float rmax = cRmax ? cRmax[idx] : 0.0f;
	const float reach = margin + rmax;          // 중심이 이보다 멀면 타원체도 안 닿는다
	for (int c = 0; c < nc; ++c) {
		const KinCollider& C = cols[c];
		const float dx = p.x - C.t[0], dy = p.y - C.t[1], dz = p.z - C.t[2];
		const float q0 = C.R[0] * dx + C.R[3] * dy + C.R[6] * dz;   // q = Rᵀ (p − t)
		const float q1 = C.R[1] * dx + C.R[4] * dy + C.R[7] * dz;
		const float q2 = C.R[2] * dx + C.R[5] * dy + C.R[8] * dz;
		float n0 = 0.0f, n1 = 0.0f, n2 = 1.0f, dist;
		if (C.type == 1) {
			const float a0 = fabsf(q0) - C.h[0], a1 = fabsf(q1) - C.h[1], a2 = fabsf(q2) - C.h[2];
			if (a0 > 0.0f || a1 > 0.0f || a2 > 0.0f) {
				const float o0 = fmaxf(a0, 0.0f), o1 = fmaxf(a1, 0.0f), o2 = fmaxf(a2, 0.0f);
				dist = sqrtf(o0 * o0 + o1 * o1 + o2 * o2);
				if (dist >= reach) continue;
				const float inv = 1.0f / fmaxf(dist, 1e-20f);
				n0 = copysignf(o0 * inv, q0); n1 = copysignf(o1 * inv, q1); n2 = copysignf(o2 * inv, q2);
			}
			else {
				if (a0 >= a1 && a0 >= a2) { dist = a0; n0 = (q0 >= 0.0f) ? 1.0f : -1.0f; n2 = 0.0f; }
				else if (a1 >= a2)        { dist = a1; n1 = (q1 >= 0.0f) ? 1.0f : -1.0f; n2 = 0.0f; }
				else                      { dist = a2; n2 = (q2 >= 0.0f) ? 1.0f : -1.0f; }
			}
		}
		else {
			float e2 = q2;
			if (C.type == 2) e2 = q2 - fminf(fmaxf(q2, -C.h[1]), C.h[1]);   // 캡슐: 축 선분까지
			const float r = sqrtf(q0 * q0 + q1 * q1 + e2 * e2);
			dist = r - C.h[0];
			if (dist >= reach) continue;
			if (r > 1e-20f) { n0 = q0 / r; n1 = q1 / r; n2 = e2 / r; }
		}
		const float wx = C.R[0] * n0 + C.R[1] * n1 + C.R[2] * n2;   // n = R · n_local
		const float wy = C.R[3] * n0 + C.R[4] * n1 + C.R[5] * n2;
		const float wz = C.R[6] * n0 + C.R[7] * n1 + C.R[8] * n2;
		if (cM && rmax > 0.0f) {                                    // 타원체가 법선 쪽으로 튀어나온 만큼 더 가깝다
			dist -= contactSupport(&cM[6 * idx], wx, wy, wz);
			if (dist >= margin) continue;
		}
		const float corr = margin - dist;
		const float3 pIn = p;
		p.x += wx * corr; p.y += wy * corr; p.z += wz * corr;
		float3 dp = make_float3(wx * corr, wy * corr, wz * corr);
		if (mu > 0.0f) {
			// 로컬 점 q 가 직전 자세에서 있던 곳 → 이번 스텝에 표면이 움직인 양 dCol = pIn − (Rp q + tp)
			const float ox = C.Rp[0] * q0 + C.Rp[1] * q1 + C.Rp[2] * q2 + C.tp[0];
			const float oy = C.Rp[3] * q0 + C.Rp[4] * q1 + C.Rp[5] * q2 + C.tp[1];
			const float oz = C.Rp[6] * q0 + C.Rp[7] * q1 + C.Rp[8] * q2 + C.tp[2];
			const float3 x0 = posPrev[idx];
			const float rx = (p.x - x0.x) - (pIn.x - ox), ry = (p.y - x0.y) - (pIn.y - oy), rz = (p.z - x0.z) - (pIn.z - oz);
			const float rn = rx * wx + ry * wy + rz * wz;
			const float tx = rx - rn * wx, ty = ry - rn * wy, tz = rz - rn * wz;   // 표면에 대한 접선 미끄러짐
			const float lt = sqrtf(tx * tx + ty * ty + tz * tz);
			if (lt > 1e-20f) {
				const float k = fminf(1.0f, mu * corr / lt);                       // 정지 마찰이면 전부, 아니면 μ·깊이 만큼
				p.x -= k * tx; p.y -= k * ty; p.z -= k * tz;
				dp.x -= k * tx; dp.y -= k * ty; dp.z -= k * tz;
			}
		}
		moved = true;
		// 반작용용: 밀어낸 변위 합 + 접촉 중심(밀어낸 크기로 가중한 위치) — 충격량을 그 점에 주면 토크도 생긴다
		float* acc = &push[KINCOL_ACC * c];
		const float w = sqrtf(dp.x * dp.x + dp.y * dp.y + dp.z * dp.z);
		atomicAdd(&acc[0], dp.x);
		atomicAdd(&acc[1], dp.y);
		atomicAdd(&acc[2], dp.z);
		atomicAdd(&acc[3], w * p.x);
		atomicAdd(&acc[4], w * p.y);
		atomicAdd(&acc[5], w * p.z);
		atomicAdd(&acc[6], w);
		if (countHits) atomicAdd(&hits[c], 1);
	}
	if (moved) pos[idx] = p;
}

// 속도 단계: xpbdUpdateVelocityKernel의 안정 경로와 같고, 바닥에 닿은 입자에만 반발·마찰을 준다.
// 닿지 않은 입자의 산술은 기존 커널과 완전히 같다.
//   vPre : 이 커널에 들어올 때 vel[]에 남아 있는 predict 직후 속도 (중력 포함)
//   반발 : 다가오던 법선속도 vnPre < 0 → vn ≥ −e·vnPre.  |vnPre| ≤ restingSpeed(= 2g·dt)면 e = 0
//          (정지 접촉에서 매 프레임 g·dt만큼 튀어 떨리는 것을 막는다)
//   마찰 : 바닥이 준 법선 충격 jn = −(1+e)·vnPre 에 비례해 접선속도를 μ·jn만큼 깎는다 (쿨롱).
//          정지 접촉이면 jn = g·dt라 감속도가 정확히 μg.
// CPU 검산(단일 입자, dt=1/60): 반등 정점/e² 0.93~0.97, 미끄럼 정지거리/(v0²/2μg) 1.03~1.05, 정지 떨림 0.
__global__ void xpbdUpdateVelocityGroundKernel(
	int N,
	const float3* pos_curr,
	const float3* pos_pred_final,
	float3* vel,
	float3* pos_curr_out,
	const float* invMass,
	float dt,
	float velDamping,
	float3 n,
	float planeD,
	float slop,
	float friction,
	float restitution,
	float restingSpeed,
	const float* rg)   // 타원체 접촉: 입자별 법선 방향 반경 (nullptr = 점)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	const float3 x = pos_curr[idx];
	const float3 p = pos_pred_final[idx];
	const float w = invMass[idx];
	if (w <= 0.0f) {
		vel[idx] = make_float3(0.0f, 0.0f, 0.0f);
		pos_curr_out[idx] = p;
		return;
	}
	const float3 vPre = vel[idx];
	const float3 dx = f3_sub(p, x);
	const float dx2 = f3_len2(dx);
	if (!isfinite(dx2)) {
		vel[idx] = make_float3(0.0f, 0.0f, 0.0f);
		pos_curr_out[idx] = p;
		return;
	}
	// 이동량이 0이어도 접촉 응답은 계산한다. 이미 바닥에 붙어 있던 입자가 아래로 밀렸다가 투영으로
	// 제자리에 돌아오면 dx = 0 이지만 다가오던 속도 vPre 는 있다 — 예전에는 여기서 조기 반환해 반발이 사라졌다
	// (외부 검토 지적, 2026-09-16). 닿지 않은 입자는 아래 마지막 검사에서 예전과 같은 0이 된다.
	float3 v = (dx2 < 1e-18f) ? make_float3(0.0f, 0.0f, 0.0f) : f3_mul(dx, 1.0f / fmaxf(dt, 1e-8f));

	const float gap = n.x * p.x + n.y * p.y + n.z * p.z - planeD - (rg ? rg[idx] : 0.0f);
	if (gap <= slop) {
		const float vn = n.x * v.x + n.y * v.y + n.z * v.z;
		const float vnPre = n.x * vPre.x + n.y * vPre.y + n.z * vPre.z;
		float vnNew = vn;
		float jn = 0.0f;
		if (vnPre < 0.0f) {
			const float e = (-vnPre <= restingSpeed) ? 0.0f : restitution;
			vnNew = fmaxf(vn, -e * vnPre);
			jn = -(1.0f + e) * vnPre;
		}
		float3 vt = make_float3(v.x - vn * n.x, v.y - vn * n.y, v.z - vn * n.z);
		const float vtLen = f3_len(vt);
		const float dvt = friction * jn;
		if (vtLen <= dvt || vtLen < 1e-12f) {
			vt = make_float3(0.0f, 0.0f, 0.0f);   // 정지 마찰: 접선 미끄럼 제거
		}
		else {
			vt = f3_mul(vt, (vtLen - dvt) / vtLen);
		}
		v = make_float3(vt.x + vnNew * n.x, vt.y + vnNew * n.y, vt.z + vnNew * n.z);
	}

	const float damping = fminf(fmaxf(velDamping, 0.0f), 1.0f);
	v = f3_mul(v, 1.0f - damping);
	if (!isfinite(v.x) || !isfinite(v.y) || !isfinite(v.z) || f3_len2(v) < 1e-16f) {
		v = make_float3(0.0f, 0.0f, 0.0f);
	}
	vel[idx] = v;
	pos_curr_out[idx] = p;
}

// 이웃 그래프는 더 이상 재포장하지 않는다. preprocessCUDA가 d_offset/d_nbrCount/d_nbrIdx를
// 직접 읽으므로, 여기서는 렌더가 필요로 하는 변형 후 위치와 도달 시간만 옮긴다.
__global__ void PackKernel(
	int N,
	const float3* pos,
	const float* time,
	float* out_means3D,
	float* out_nbr_time)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	const float3 p = pos[idx];
	out_means3D[3 * idx + 0] = p.x;
	out_means3D[3 * idx + 1] = p.y;
	out_means3D[3 * idx + 2] = p.z;

	out_nbr_time[idx] = time[idx];
}

// ===============================================================
// Volume Gaussian — 사전계산 (프로그램/그래프 로드 시 1회)
//
//   (1)  c     = (1/n) SUM_{k in C} mu_k
//   (2)  Sigma = (1/n) SUM_{k in C} (mu_k - c)(mu_k - c)^T
//   (3)  V     = (4/3) pi sqrt(det Sigma)     <- 사면체 rest 부피를 대신한다
//
// 클러스터 C(i) = {i} U N(i) 는 기존 유사도 그래프(CSR)를 그대로 쓴다.
// ===============================================================

// 대칭 3x3을 6원소로 저장: [xx, yy, zz, xy, xz, yz]
__device__ __forceinline__ void storeSym3(float* dst, const glm::mat3& S)
{
	dst[0] = S[0][0];
	dst[1] = S[1][1];
	dst[2] = S[2][2];
	dst[3] = S[1][0];
	dst[4] = S[2][0];
	dst[5] = S[2][1];
}

__device__ __forceinline__ glm::mat3 loadSym3(const float* src)
{
	glm::mat3 S(0.0f);
	S[0][0] = src[0];
	S[1][1] = src[1];
	S[2][2] = src[2];
	S[0][1] = S[1][0] = src[3];
	S[0][2] = S[2][0] = src[4];
	S[1][2] = S[2][1] = src[5];
	return S;
}

// Sigma를 스케일과 형상으로 분리한다:  Sigma = s * SigmaHat,  s = trace(Sigma)/3
//   det Sigma = s^3 * det SigmaHat
// s는 양수의 합이라 상쇄가 없고, det SigmaHat은 O(1)이라 float32로 안전하다.
// det Sigma를 직접 계산하면 성분이 ~1e-3일 때 det가 ~1e-8~1e-17까지 내려가
// 상쇄로 상대오차가 폭발한다. (2D 참조 구현은 float64라 이 문제가 드러나지 않는다.)
// 클램프 발동 카운터 (디바이스 전역). [0]=J [1]=스텝 [2]=Σ⁻¹ 고유값 하한.
// 커널이 시그니처 변경 없이 직접 atomicAdd. 호스트는 cudaMemcpy{To,From}Symbol로 접근.
__device__ unsigned int g_clampCtr[3];

// (Σ⁻¹ 축퇴 방어 방식 토글 g_dSinvIsoReg/Eps/DetThr 은 파일 상단에 선언돼 있다 —
//  호스트 setter가 그보다 앞에서 cudaMemcpyToSymbol을 호출하므로.)

__device__ __forceinline__ bool sigmaScaleAndShapeDet(
	const glm::mat3& S, float& outScale, float& outDetHat)
{
	const float s = (S[0][0] + S[1][1] + S[2][2]) * (1.0f / 3.0f);
	if (!isfinite(s) || s <= 1e-24f) return false;
	const float detHat = glm::determinant(S / s);
	if (!isfinite(detHat) || detHat <= 1e-6f) return false; // 형상이 축퇴됨
	outScale = s;
	outDetHat = detHat;
	return true;
}

// 3DGS computeCov3D의 6원소 순서 [xx, xy, xz, yy, yz, zz]를 glm::mat3로
__device__ __forceinline__ glm::mat3 loadCov3D_3dgs(const float* c)
{
	glm::mat3 M(0.0f);
	M[0][0] = c[0];
	M[0][1] = M[1][0] = c[1];
	M[0][2] = M[2][0] = c[2];
	M[1][1] = c[3];
	M[1][2] = M[2][1] = c[4];
	M[2][2] = c[5];
	return M;
}

__global__ void precomputeVolumeGaussianKernel(
	int N,
	const float3* pos_rest,
	const int* offset,
	const int* nbrCount,
	const int* nbrIdx,
	// 원본 그래프 CSR (restLen 계산용 — 모든 노드가, 반장 여부와 무관하게 갖는다)
	const int* gOffset,
	const int* gCount,
	const int* gIdx,
	// mixture rest용 렌더 속성 (useMixture=0이거나 nullptr이면 무시)
	const glm::vec3* gsScales,
	const glm::vec4* gsRotations,
	const float* gsOpacity,
	int useMixture,
	float* SigmaRest,   // [N*6]
	float* detRest,     // [N]
	float* Vrest,       // [N]
	float* restLen,     // [N] 이웃 rest 최소거리 (스텝 클램프의 길이 스케일) — 모든 노드
	int* matType,       // [N]
	float anisoThreshold)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	float* sig = SigmaRest + idx * 6;
	for (int k = 0; k < 6; ++k) sig[k] = 0.0f;
	detRest[idx] = 0.0f;
	Vrest[idx] = 0.0f;
	// [진단①] matType 코드로 탈락 사유를 구분한다. 물리는 (matType != 0)만 보므로 거동 불변.
	//   0=volume, 1=surface, 2=fiber(이방성), 3=이웃부족, 4=수치실패(정밀도)
	matType[idx] = 3; // 기본값: 이웃부족. 아래에서 진행하며 승격/재분류된다.

	// ★ restLen(스텝 클램프 길이 스케일)은 '원본 그래프 이웃'의 최소 거리로 채운다.
	//   ── 반장 여부·볼륨 클러스터 반경(k·r)과 완전히 독립. 모든 노드가 값을 갖는다. ──
	//   과거 버그: 비반장 노드는 아래 게이트에서 return 되어 restLen=0으로 남았고,
	//   Apply 커널의 스텝 클램프가 (rl>1e-12) 조건에서 통째로 건너뛰어져(클램프 없음)
	//   비반장 노드의 보정이 무제한 적용 → r≥2에서 강한 드래그 시 발산.
	{
		const float3 p0g = pos_rest[idx];
		const int goff = gOffset[idx], gcnt = gCount[idx];
		float dmin = FLT_MAX;
		for (int e = 0; e < gcnt; ++e) {
			const int j = gIdx[goff + e];
			if (j < 0 || j >= N || j == idx) continue;
			const float3 pj = pos_rest[j];
			const float dl = f3_len(f3_sub(pj, p0g));
			if (dl > 1e-12f) dmin = fminf(dmin, dl);
		}
		restLen[idx] = (dmin < FLT_MAX) ? dmin : 0.0f;
	}

	const int off = offset[idx];
	const int cnt = nbrCount[idx];
	if (cnt < 4) return; // 이웃이 3개 미만이면 3D 퍼짐 자체를 잴 수 없다.

	// (1) 무게중심 (+ 길이 스케일: 자신→멤버 평균 거리. k-ring이면 자연히 커진다)
	const float3 p0 = pos_rest[idx];
	float3 c = p0;
	int used = 0;
	for (int k = 0; k < cnt; ++k) {
		const int nIdx = nbrIdx[off + k];
		if (nIdx < 0 || nIdx >= N) continue;
		c = f3_add(c, pos_rest[nIdx]);
		++used;
	}
	if (used < 3) return;
	const float n = (float)(used + 1);
	const float invN = 1.0f / n;
	c = f3_mul(c, invN);
	// restLen은 위에서 원본 그래프로 이미 채웠다(반장/클러스터 반경과 무관).

	// (2) 공분산
	glm::mat3 S(0.0f);
	{
		const float3 d = f3_sub(pos_rest[idx], c);
		const glm::vec3 v(d.x, d.y, d.z);
		S += glm::outerProduct(v, v);
	}
	for (int k = 0; k < cnt; ++k) {
		const int nIdx = nbrIdx[off + k];
		if (nIdx < 0 || nIdx >= N) continue;
		const float3 d = f3_sub(pos_rest[nIdx], c);
		const glm::vec3 v(d.x, d.y, d.z);
		S += glm::outerProduct(v, v);
	}
	S *= invN;

	// (3) 수치적으로 부피를 정의할 수 있는 클러스터인지 먼저 거른다.
	// 예전에는 det를 1e-18로 하한 클램프했는데, 그러면 rest 상태인데도
	// detRest(클램프됨) != detCur(raw) 가 되어 J가 1에서 크게 벗어난다. 클램프는 답이 아니다.
	// ★ 이 게이트는 '중심점 공분산' 기준이어야 한다 — 런타임 저울(J)이 중심점만 쓰므로,
	//   저울이 성립하지 않는 클러스터는 mixture가 아무리 통통해도 제약을 걸 수 없다.
	float sRest, detHatRest;
	if (!sigmaScaleAndShapeDet(S, sRest, detHatRest)) { matType[idx] = 4; return; } // [진단①] 4=수치실패

	// (3.5) 명찰/신분증용 mixture 공분산: Σ_mix = Σ_centers + B
	// B = 멤버 가우시안들이 '자신의 몸으로 차지하는 퍼짐'(R·S²·Rᵀ)의 opacity 가중 평균.
	// 총분산의 법칙: 그물로 감싼 전체 퍼짐 = 중심들의 퍼짐 + 각자 몸집의 평균.
	// ★ 저울(detRest)에는 절대 섞지 않는다 — rest에서 J=1이 깨진다.
	glm::mat3 Smix = S;
	if (useMixture && gsScales != nullptr && gsRotations != nullptr) {
		glm::mat3 B(0.0f);
		float wSum = 0.0f;
		{
			const float w = gsOpacity ? fmaxf(gsOpacity[idx], 1e-3f) : 1.0f;
			float cov6[6];
			computeCov3D(gsScales[idx], 1.0f, gsRotations[idx], cov6);
			B += w * loadCov3D_3dgs(cov6);
			wSum += w;
		}
		for (int k = 0; k < cnt; ++k) {
			const int nIdx = nbrIdx[off + k];
			if (nIdx < 0 || nIdx >= N) continue;
			const float w = gsOpacity ? fmaxf(gsOpacity[nIdx], 1e-3f) : 1.0f;
			float cov6[6];
			computeCov3D(gsScales[nIdx], 1.0f, gsRotations[nIdx], cov6);
			B += w * loadCov3D_3dgs(cov6);
			wSum += w;
		}
		if (wSum > 1e-12f) {
			B *= (1.0f / wSum);
			Smix = S + B;
		}
	}

	// (4) 물질 타입 분류 — 고유값 이방성 (mixture 기준: 중심이 평면에 깔려 있어도
	// 가우시안 몸집이 두께를 채우면 volume으로 올바르게 승격된다)
	glm::vec3 eig;
	glm::mat3 V;
	eigenDecomposition_glm(Smix, eig, V); // 오름차순 정렬됨
	const float lmax = fmaxf(eig.z, 1e-24f);
	const float a1 = eig.x / lmax; // 최소축 / 최대축
	const float a2 = eig.y / lmax;

	if (a1 > anisoThreshold)      matType[idx] = 0; // 3D로 퍼짐  → 볼륨
	else if (a2 > anisoThreshold) matType[idx] = 1; // 평면에 깔림 → 표면
	else                          matType[idx] = 2; // 직선에 놓임 → 섬유

	// (5) 저장
	storeSym3(sig, S);
	// 저울의 분모: 반드시 중심점 공분산의 det (런타임 detCur와 같은 경로)
	const float d = sRest * sRest * sRest * detHatRest; // = det(S), 안정하게 재구성
	detRest[idx] = d;
	// 명찰: mixture 부피 — "물질이 실제로 차지하는 공간"에 가깝게.
	// compliance 앵커 α = c/(λ·V_rest) 의 V로만 쓰이고, J 측정에는 관여하지 않는다.
	float volDet = d;
	if (useMixture) {
		float sM, detHatM;
		if (sigmaScaleAndShapeDet(Smix, sM, detHatM))
			volDet = sM * sM * sM * detHatM;
	}
	Vrest[idx] = (4.0f / 3.0f) * 3.14159265f * sqrtf(fmaxf(volDet, 0.0f)); // B = 가우시안 몸집의 opacity 
}

// ============================================================================
// Experimental Gaussian-cluster Stable Neo-Hookean XPBD
//
// This path is intentionally separate from the existing covariance-volume and
// shape-matching kernels. A volume-cluster acts as one mesh-free element. Its
// best-fit affine deformation gradient is
//
//     F = (sum q p^T / n) (sum p p^T / n)^-1,
//
// where p and q are rest/current offsets from their respective centroids.
// Hydrostatic and distortional constraints follow Macklin & Mueller (2021):
//
//     C_H = det(F) - gamma,  gamma = 1 + mu/lambda
//     C_D = sqrt(trace(F^T F)).
//
// The two constraints are projected as a coupled 2x2 XPBD block. This retains
// their rest-state force balance better than two unrelated Jacobi projections.
// Corrections are written to a private per-cluster blackboard and applied with
// the already-built reverse CSR, so legacy solver buffers remain untouched.
// ============================================================================

__device__ __forceinline__ bool finiteMat3GNH(const glm::mat3& M)
{
	for (int c = 0; c < 3; ++c)
		for (int r = 0; r < 3; ++r)
			if (!isfinite(M[c][r])) return false;
	return true;
}

__device__ __forceinline__ glm::mat3 cofactorMat3GNH(const glm::mat3& F)
{
	// GLM matrices are column-major. For F=[a b c], cof(F) has columns
	// b x c, c x a, a x b and satisfies d det(F) = cof(F) : dF.
	glm::mat3 C(0.0f);
	C[0] = glm::cross(F[1], F[2]);
	C[1] = glm::cross(F[2], F[0]);
	C[2] = glm::cross(F[0], F[1]);
	return C;
}

__device__ __forceinline__ void storeMat3GNH(float* dst, const glm::mat3& M)
{
	for (int c = 0; c < 3; ++c)
		for (int r = 0; r < 3; ++r)
			dst[c * 3 + r] = M[c][r];
}

__device__ __forceinline__ glm::mat3 loadMat3GNH(const float* src)
{
	glm::mat3 M(0.0f);
	for (int c = 0; c < 3; ++c)
		for (int r = 0; r < 3; ++r)
			M[c][r] = src[c * 3 + r];
	return M;
}

__global__ void precomputeGaussianNHRestKernel(
	int N,
	const float3* posRest,
	const int* offset,
	const int* count,
	const int* members,
	const float* sigmaRest,
	const int* matType,
	float3* restC,
	float* restSinv,
	unsigned char* valid)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;
	valid[idx] = 0;
	restC[idx] = make_float3(0.0f, 0.0f, 0.0f);
	for (int k = 0; k < 6; ++k) restSinv[idx * 6 + k] = 0.0f;
	if (matType[idx] != 0) return;

	const int off = offset[idx];
	const int cnt = count[idx];
	float3 c = posRest[idx];
	int used = 0;
	for (int k = 0; k < cnt; ++k) {
		const int j = members[off + k];
		if (j < 0 || j >= N) continue;
		c = f3_add(c, posRest[j]);
		++used;
	}
	if (used < 3) return;
	c = f3_mul(c, 1.0f / (float)(used + 1));

	const glm::mat3 P = loadSym3(sigmaRest + idx * 6);
	const float s = (P[0][0] + P[1][1] + P[2][2]) * (1.0f / 3.0f);
	if (!isfinite(s) || s <= 1.0e-24f) return;
	glm::mat3 invHat;
	if (!inverse3x3_safe(P / s, invHat, 1.0e-9f)) return;
	const glm::mat3 Pinv = invHat / s;
	if (!finiteMat3GNH(Pinv)) return;

	restC[idx] = c;
	storeSym3(restSinv + idx * 6, Pinv);
	valid[idx] = 1;
}

__global__ void xpbdGaussianNHClusterSolveKernel(
	int N,
	const float3* pos,
	const float3* posRest,
	const float* invMass,
	const int* offset,
	const int* count,
	const int* members,
	const float3* restC,
	const float* restSinv,
	const unsigned char* valid,
	const float* restVolume,
	float* lambdaD,
	float* lambdaH,
	float young,
	float poisson,
	float complianceScale,
	float dt,
	float invMassScale,
	float* clusterF,
	float* coefD,
	float* coefH)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;
	coefD[idx] = 0.0f;
	coefH[idx] = 0.0f;
	if (!valid[idx]) return;

	const int off = offset[idx];
	const int cnt = count[idx];
	float3 c = pos[idx];
	int used = 0;
	for (int k = 0; k < cnt; ++k) {
		const int j = members[off + k];
		if (j < 0 || j >= N) continue;
		c = f3_add(c, pos[j]);
		++used;
	}
	if (used < 3) return;
	const float invN = 1.0f / (float)(used + 1);
	c = f3_mul(c, invN);

	const float3 c0 = restC[idx];
	glm::mat3 A(0.0f);
	{
		const float3 pf = f3_sub(posRest[idx], c0);
		const float3 qf = f3_sub(pos[idx], c);
		A += glm::outerProduct(glm::vec3(qf.x, qf.y, qf.z), glm::vec3(pf.x, pf.y, pf.z));
	}
	for (int k = 0; k < cnt; ++k) {
		const int j = members[off + k];
		if (j < 0 || j >= N) continue;
		const float3 pf = f3_sub(posRest[j], c0);
		const float3 qf = f3_sub(pos[j], c);
		A += glm::outerProduct(glm::vec3(qf.x, qf.y, qf.z), glm::vec3(pf.x, pf.y, pf.z));
	}
	A *= invN;
	const glm::mat3 Pinv = loadSym3(restSinv + idx * 6);
	const glm::mat3 F = A * Pinv;
	if (!finiteMat3GNH(F)) return;

	float i1 = 0.0f;
	for (int cc = 0; cc < 3; ++cc)
		for (int rr = 0; rr < 3; ++rr)
			i1 += F[cc][rr] * F[cc][rr];
	if (!isfinite(i1) || i1 <= 1.0e-16f || i1 > 1.0e16f) return;
	const float normF = sqrtf(i1);
	const float detF = glm::determinant(F);
	if (!isfinite(detF) || fabsf(detF) > 1.0e8f) return;
	const glm::mat3 cofF = cofactorMat3GNH(F);
	if (!finiteMat3GNH(cofF)) return;

	const float nu = fminf(fmaxf(poisson, 1.0e-4f), 0.49f);
	const float E = fmaxf(young, 1.0f);
	const float mu = E / (2.0f * (1.0f + nu));
	const float lame = E * nu / ((1.0f + nu) * (1.0f - 2.0f * nu));
	const float gamma = 1.0f + mu / lame;
	const float V0 = restVolume[idx];
	if (!isfinite(V0) || V0 <= 1.0e-20f) return;

	const float CD = normF;
	const float CH = detF - gamma;
	float kDD = 0.0f, kHH = 0.0f, kHD = 0.0f;

	// Both gradients use rest offsets. Center derivatives cancel because the
	// uniformly weighted rest offsets sum to zero.
	{
		const float3 pf = f3_sub(posRest[idx], c0);
		const glm::vec3 r = Pinv * glm::vec3(pf.x, pf.y, pf.z);
		const glm::vec3 gD = (invN / normF) * (F * r);
		const glm::vec3 gH = invN * (cofF * r);
		const float w = invMass[idx] * invMassScale;
		kDD += w * glm::dot(gD, gD);
		kHH += w * glm::dot(gH, gH);
		kHD += w * glm::dot(gD, gH);
	}
	for (int k = 0; k < cnt; ++k) {
		const int j = members[off + k];
		if (j < 0 || j >= N) continue;
		const float3 pf = f3_sub(posRest[j], c0);
		const glm::vec3 r = Pinv * glm::vec3(pf.x, pf.y, pf.z);
		const glm::vec3 gD = (invN / normF) * (F * r);
		const glm::vec3 gH = invN * (cofF * r);
		const float w = invMass[j] * invMassScale;
		kDD += w * glm::dot(gD, gD);
		kHH += w * glm::dot(gH, gH);
		kHD += w * glm::dot(gD, gH);
	}

	const float dt2 = fmaxf(dt * dt, 1.0e-8f);
	const float scale = fmaxf(complianceScale, 1.0e-8f);
	const float alphaD = scale / (mu * V0 * dt2);
	const float alphaH = scale / (lame * V0 * dt2);
	const float a = kHH + alphaH;
	const float b = kHD;
	const float d = kDD + alphaD;
	const float rhsH = -CH - alphaH * lambdaH[idx];
	const float rhsD = -CD - alphaD * lambdaD[idx];
	const float detK = a * d - b * b;
	if (!isfinite(detK) || detK <= 1.0e-20f) return;

	const float dLamH = (rhsH * d - b * rhsD) / detK;
	const float dLamD = (a * rhsD - b * rhsH) / detK;
	if (!isfinite(dLamH) || !isfinite(dLamD)) return;
	lambdaH[idx] += dLamH;
	lambdaD[idx] += dLamD;

	storeMat3GNH(clusterF + idx * 9, F);
	coefD[idx] = dLamD * invN / normF;
	coefH[idx] = dLamH * invN;
}

__global__ void xpbdGaussianNHGatherApplyKernel(
	int N,
	const float3* posIn,
	float3* posOut,
	const float3* posRest,
	const float* invMass,
	const int* revOffset,
	const int* revCount,
	const int* revIdx,
	const float3* restC,
	const float* restSinv,
	const unsigned char* valid,
	const float* clusterF,
	const float* coefD,
	const float* coefH,
	const float* restLen,
	float invMassScale,
	float underRelax)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;
	const float3 pi = posIn[idx];
	const float wi = invMass[idx] * invMassScale;
	if (wi <= 0.0f) { posOut[idx] = pi; return; }

	const int off = revOffset[idx];
	const int cnt = revCount[idx];
	glm::vec3 sum(0.0f);
	int received = 0;
	for (int k = 0; k < cnt; ++k) {
		const int l = revIdx[off + k];
		if (l < 0 || l >= N || !valid[l]) continue;
		const float cd = coefD[l];
		const float ch = coefH[l];
		if (cd == 0.0f && ch == 0.0f) continue;

		const float3 pf = f3_sub(posRest[idx], restC[l]);
		const glm::mat3 Pinv = loadSym3(restSinv + l * 6);
		const glm::vec3 r = Pinv * glm::vec3(pf.x, pf.y, pf.z);
		const glm::mat3 F = loadMat3GNH(clusterF + l * 9);
		const glm::mat3 cofF = cofactorMat3GNH(F);
		const glm::vec3 dp = wi * (cd * (F * r) + ch * (cofF * r));
		if (!isfinite(dp.x) || !isfinite(dp.y) || !isfinite(dp.z)) continue;
		sum += dp;
		++received;
	}

	if (received <= 0) { posOut[idx] = pi; return; }
	glm::vec3 avg = sum * (1.0f / (float)received);
	const float rl = restLen[idx];
	if (rl > 1.0e-12f) {
		const float maxStep = 0.75f * rl;
		const float len = glm::length(avg);
		if (len > maxStep && len > 1.0e-20f) avg *= (maxStep / len);
	}
	const glm::vec3 out = glm::vec3(pi.x, pi.y, pi.z) + fminf(fmaxf(underRelax, 0.0f), 2.0f) * avg;
	posOut[idx] = make_float3(out.x, out.y, out.z);
}

// ===============================================================
// Volume Gaussian — 부피 제약 (매 프레임, 2패스 Jacobi)
//
//   (6)  J     = sqrt(det Sigma_cur / det Sigma_rest)   ( = |det F| )
//   (7)  C_vol = J - 1
//   (8)  grad_k C = (J/n) * Sigma_cur^-1 * (x_k - c)
//
// grad 방향이 타원체 법선이라, 눌린 축은 저항이 크고 자유로운 축으로 밀린다.
// 그래서 등방적으로 부풀지 않고 '옆으로 삐져나오는' 비압축성 거동이 나온다.
//
// 패스 1 (Accumulate): 각 클러스터가 XPBD Eq.17의 Δx_k = w_k·Δλ·∇_k C 를
//   '모든 멤버'에게 atomicAdd로 산란한다.
//   ★ 자신만 움직이면 안 된다 — 자신은 자기 클러스터의 중심이라 x_i−c ≈ 0 이고,
//     중심점 하나의 이동은 공분산(det)을 거의 못 바꾼다. 그러면 C가 줄지 않고
//     λ만 쌓여 매 프레임 계속 밀린다 → 진동/구조 붕괴 (k가 클수록 폭주).
//     전 멤버를 무게중심 기준으로 밀어야 진짜 팽창/수축이 되어 J가 반응하고 수렴한다.
//     (Region Balloon이 부드럽게 동작한 이유가 정확히 이것 — 전 멤버 적용)
// 패스 2 (Apply): 겹치는 여러 클러스터가 준 보정을 '평균'해서 적용한다.
//   (합산하면 겹침 수(~수십)만큼 과잉 보정 → 폭주. 각도 제약과 같은 방식)
// ===============================================================
__global__ void xpbdVolumeAccumulateKernel(
	int N,
	const float3* pos_pred_in,
	const float* invMass,
	const int* offset,
	const int* nbrCount,
	const int* nbrIdx,
	const float* detRest,
	const int* matType,
	float* lambda_vol,
	const float* alpha_vol,   // nullptr이면 alphaConst 사용 (Step 1: 손 튜닝 상수)
	float alphaConst,
	float dt,
	float invMassScale,
	float3* dp_sum,
	int* dp_count)
{
	// ┌──────────────────────────────────────────────────────────────────┐
	// │ 비유: "나(idx)는 우리 반의 반장이다."                              │
	// │                                                                  │
	// │  · 나 + 내 이웃 k명 = 우리 반 (클러스터)                          │
	// │  · 우리 반이 운동장에서 차지한 '자리의 넓이'를 지키는 게 내 임무    │
	// │  · 반장은 N명 전원이다. 즉 이 커널은 모든 가우시안에 대해          │
	// │    "자기 반"을 하나씩 만들고, 반들끼리는 서로 겹친다.              │
	// │    (철수는 내 반의 반원이면서, 동시에 자기 반의 반장이다)          │
	// └──────────────────────────────────────────────────────────────────┘
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	const float3 pi = pos_pred_in[idx];   // 반장인 나의 '지금' 위치

	// 게이트: 우리 반이 3차원으로 뭉쳐 있는 '살덩이 반'일 때만 부피를 지킨다.
	// 종잇장처럼 납작한 반(surface)이나 실처럼 일렬로 선 반(fiber)은
	// "부피를 지켜라"가 물리적으로 말이 안 되므로 반장 노릇을 하지 않는다.
	// (단, 그런 애들도 '옆 반의 반원'으로서는 밀려날 수 있다 — Step 6 참고)
	if (matType[idx] != 0) return;

	const int off = offset[idx];      // 우리 반 명단이 시작하는 위치
	const int cnt = nbrCount[idx];    // 우리 반 인원수(나 제외)

	// ── Step 1: 우리 반의 '한가운데'와 '퍼진 모양'을 잰다 (식 1, 2) ────────
	//
	// [1-A] 한가운데(무게중심 c) 구하기 = "다 같이 모여봐, 중심이 어디야?"
	//   나 포함 전원의 위치를 더해서 인원수로 나눈다. 그냥 평균 위치다.
	//
	//   ※ 왜 중심이 필요한가: 반 전체가 오른쪽으로 10m 걸어가도 '모양'은
	//     그대로다. 중심을 빼고 봐야 이동(translation)에 속지 않고
	//     순수하게 "얼마나 퍼졌나"만 볼 수 있다.
	//
	//   ※ 유효 이웃 판정(j<0 || j>=N 걸러내기)을 precompute와 '똑같이' 해야
	//     인원수 n이 일치한다. n이 다르면 Σ에 곱해지는 1/n이 달라져서
	//     rest 상태인데도 J가 1이 아니게 된다.
	float3 c = pi;
	int used = 0;
	for (int k = 0; k < cnt; ++k) {
		const int j = nbrIdx[off + k];
		if (j < 0 || j >= N) continue;
		c = f3_add(c, pos_pred_in[j]);   // 반원 한 명씩 위치를 더하고
		++used;
	}
	if (used < 3) return;                // 3명도 안 되면 3D 모양을 잴 수 없다 → 포기
	const float n = (float)(used + 1);   // +1 = 반장인 나
	const float invN = 1.0f / n;
	c = f3_mul(c, invN);                 // 나눠서 평균 = 우리 반 한가운데

	// [1-B] 퍼진 모양(공분산 Scur) 구하기
	//   = "각자 중심에서 어느 쪽으로 얼마나 떨어져 있니?"를 전부 모은 표
	//
	//   각 반원의 '중심 기준 상대 위치' d = (내 위치 − 중심)을 구하고,
	//   d 와 d 를 바깥곱(outerProduct)해서 더한다.
	//     outerProduct(d,d) = ⎡dx·dx  dx·dy  dx·dz⎤
	//                         ⎢dy·dx  dy·dy  dy·dz⎥   ← 3×3 대칭 행렬
	//                         ⎣dz·dx  dz·dy  dz·dz⎦
	//
	//   대각선(dx², dy², dz²)은 "x방향으로 얼마나 퍼졌나, y로는, z로는"
	//   나머지는 "x와 y가 같이 기울어져 있나"(대각선 방향 쏠림).
	//   전부 더해서 인원수로 나누면 → 우리 반을 감싸는 '타원체' 하나가 나온다.
	//   이 타원체가 바로 '부피 가우시안'이다. 단위: [m²] (길이의 제곱)
	glm::mat3 Scur(0.0f);
	{
		const float3 d = f3_sub(pi, c);      // 반장인 나부터
		const glm::vec3 v(d.x, d.y, d.z);
		Scur += glm::outerProduct(v, v);
	}
	for (int k = 0; k < cnt; ++k) {
		const int j = nbrIdx[off + k];
		if (j < 0 || j >= N) continue;
		const float3 d = f3_sub(pos_pred_in[j], c);   // 반원 j의 중심 기준 위치
		const glm::vec3 v(d.x, d.y, d.z);
		Scur += glm::outerProduct(v, v);
	}
	Scur *= invN;   // 평균 내면 완성

	// ── Step 2: "우리 반 자리가 원래보다 넓어졌나 좁아졌나?" (식 6, 7) ─────
	//
	// 타원체의 부피는 det(Σ)의 제곱근에 비례한다. 그래서
	//   J = √(지금 det / 원래 det)  = 부피가 몇 배가 되었나 (무차원)
	//     J = 1.0  → 딱 원래대로 (할 일 없음)
	//     J = 0.8  → 20% 눌렸다   (넓혀야 함)
	//     J = 1.3  → 30% 부풀었다 (좁혀야 함)
	//
	// 이게 FEM의 det(F)와 정확히 같은 값이다. 증명:
	//   변형 F가 가해지면 Σ' = F Σ Fᵀ  ⟹  det Σ' = (det F)²·det Σ
	//   ⟹ √(det Σ' / det Σ) = |det F|      ← 근사가 아니라 등식
	//
	// ※ 왜 sigmaScaleAndShapeDet 을 쓰나 (그냥 determinant 쓰면 안 되나):
	//   Σ 성분이 ~1e-3 이면 det는 ~1e-8~1e-17까지 내려간다. float32는 유효숫자
	//   7자리뿐인데 3×3 det는 비슷한 크기끼리 '빼는' 연산이라 상쇄로 오차가 폭발한다.
	//   그래서 Σ = s·Σ̂ (s = trace/3) 로 크기와 모양을 분리해서
	//   det Σ = s³ · det Σ̂ 로 재조립한다. s는 덧셈뿐이라 안전하고 det Σ̂는 O(1)이다.
	//   ★ 핵심: precompute(원래 det)와 여기(지금 det)가 '똑같은 경로'로 계산돼야
	//     rest 상태에서 두 값이 정확히 같아져 J = 1이 나온다. (rest-J self-check로 검증)
	float sCur, detHatCur;
	if (!sigmaScaleAndShapeDet(Scur, sCur, detHatCur)) return;
	const float detCur = sCur * sCur * sCur * detHatCur;   // = det(Scur), 안전하게
	const float dRest = detRest[idx];                      // 처음에 재둔 '원래' 넓이
	if (dRest < 1e-30f) return;

	float J = sqrtf(detCur / dRest); // = |det F|, 무차원
	if (!isfinite(J) || J <= 0.0f) return;
	// [안전판] J 하드 클램프 (gather 커널과 동일, 극단값 방어).
	if (J > 3.0f || J < 0.333f) { atomicAdd(&g_clampCtr[0], 1u); J = fminf(3.0f, fmaxf(0.333f, J)); }

	const float C = J - 1.0f;   // 제약 위반량. 0으로 만드는 게 목표.
	// 데드밴드: 0.01%보다 작은 차이는 계산 오차 수준이라 무시한다.
	// (안 그러면 가만히 있는 물체를 노이즈로 계속 흔들게 된다)
	if (fabsf(C) < 1e-4f) return;

	// ── Step 3: "어느 방향으로 밀어야 효율적인가?" (Σ의 역행렬) ───────────
	//
	// 그냥 사방으로 똑같이 부풀리면 안 된다. 우리 반이 이미 옆으로 납작하게
	// 눌려 있다면, 눌린 위/아래로 미는 건 힘들고 옆으로 미는 게 훨씬 쉽다.
	// Σ⁻¹ 이 그 '쉬운 방향'을 알려준다.
	//   · 이미 많이 퍼진 축 → Σ 값 큼 → Σ⁻¹ 값 작음 → 살살 민다
	//   · 눌려서 좁아진 축 → Σ 값 작음 → Σ⁻¹ 값 큼   → 강하게 저항/민다
	// 이 방향이 기하학적으로 '타원체 표면의 법선'이고,
	// 그래서 누르면 등방적으로 부푸는 게 아니라 '옆으로 삐져나오는' 거동이 나온다.
	//
	// [G4] 안전장치: 반이 완전히 납작해지면(det→0) Σ⁻¹이 무한대로 폭발해
	//   엉뚱한 방향으로 날아간다. 그래서 역행렬을 구하기 '직전에만'
	//   가장 짧은 축을 가장 긴 축의 6% 이상으로 억지로 늘려준다.
	//   ※ J는 위에서 이미 raw 값으로 쟀으므로 '측정값'은 오염되지 않는다.
	//     여기서 손대는 건 '미는 방향' 계산용 사본뿐이다.
	//   ※ 모드 1(등방 정규화)이면 고유분해 없이 대각선에 같은 값을 더해 같은 목적을 달성한다.
	//     ~600 FLOP → ~15 FLOP. 자세한 근거는 g_dSinvIsoReg 선언부 주석 참조.
	glm::mat3 Sinv;
	if (g_dSinvIsoReg) {
		glm::mat3 Sreg = Scur;
		if (detHatCur < g_dSinvIsoDetThr) {          // 축퇴한 클러스터만
			atomicAdd(&g_clampCtr[2], 1u);
			const float d = g_dSinvIsoEps * sCur;    // sCur = trace/3 (Step 2에서 계산됨)
			Sreg[0][0] += d; Sreg[1][1] += d; Sreg[2][2] += d;
		}
		if (!inverse3x3_safe(Sreg, Sinv, 1e-20f)) return;
	}
	else {
		glm::vec3 eig;   // 타원체 세 축의 길이²  (오름차순: eig.x=가장 짧은 축)
		glm::mat3 V;     // 세 축이 향하는 방향
		eigenDecomposition_glm(Scur, eig, V);
		const float lmax = fmaxf(eig.z, 1e-20f);      // 가장 긴 축
		const float lmin = 0.06f * lmax;              // 하한선 = 긴 축의 6%
		if (eig.x < lmin || eig.y < lmin) atomicAdd(&g_clampCtr[2], 1u); // G4 실제 발동
		const glm::vec3 e = glm::max(eig, glm::vec3(lmin));  // 너무 짧은 축은 끌어올림

		// 손본 축 길이로 행렬을 재조립: Σ_clamped = V·D·Vᵀ
		glm::mat3 D(0.0f);
		D[0][0] = e.x; D[1][1] = e.y; D[2][2] = e.z;
		const glm::mat3 Sclamped = V * D * glm::transpose(V);

		if (!inverse3x3_safe(Sclamped, Sinv, 1e-20f)) return;
	}

	// ── Step 4: "각자 어느 쪽으로 움직여야 하나?" (식 8) ──────────────────
	//
	//   g_k = (J/n) · Σ⁻¹ · (x_k − c)      단위: [1/m]
	//          ↑         ↑      ↑
	//          │         │      └ 중심에서 나를 향하는 화살표 ("나는 중심 기준 왼쪽!")
	//          │         └ 그 화살표를 '쉬운 방향'으로 휘어줌 (Step 3)
	//          └ 전체 세기 조절 (n으로 나눠서 인원수 많다고 과해지지 않게)
	//
	// 비유: 반원들이 전부 중심을 바라보고 서 있다. 반이 눌렸으면(J<1)
	//   "다들 중심에서 바깥으로 한 발짝!" — 왼쪽에 선 애는 더 왼쪽으로,
	//   오른쪽에 선 애는 더 오른쪽으로. 각자 자기가 선 방향으로 벌어진다.
	//   부풀었으면(J>1) 반대로 안쪽으로 모인다.
	//
	// 여기서는 아직 '움직이지 않는다'. 두 가지만 계산한다:
	//   (a) g_self  — 반장인 내 화살표 (Step 6에서 쓸 것)
	//   (b) denom   — 반 전체의 '반응 총합'. 다음 Step에서 세기를 정하는 데 필요.
	//
	// denom = Σ_k w_k·|g_k|²  의 의미:
	//   "모두가 이 방향으로 밀면 부피가 얼마나 빨리 변하나?"
	//   무거운 애(w 작음)만 있으면 잘 안 변하고 → denom 작음 → 더 세게 밀어야 함.
	//   ★ 주의: denom은 '전원'에 대해 더한다. XPBD 이론이 전원이 움직인다고
	//     가정하고 유도된 식이기 때문. (그래서 Step 6도 전원에게 적용해야 앞뒤가 맞는다)
	const float gs = J * invN;    // (J/n) 공통 계수
	float denom = 0.0f;
	float3 g_self;
	{
		const float3 d = f3_sub(pi, c);                       // 중심 → 나
		const glm::vec3 g = gs * (Sinv * glm::vec3(d.x, d.y, d.z));
		g_self = make_float3(g.x, g.y, g.z);                  // 내 화살표 저장
		denom += (invMass[idx] * invMassScale) * f3_len2(g_self);
	}
	for (int k = 0; k < cnt; ++k) {
		const int j = nbrIdx[off + k];
		if (j < 0 || j >= N) continue;
		const float3 d = f3_sub(pos_pred_in[j], c);           // 중심 → 반원 j
		const glm::vec3 g = gs * (Sinv * glm::vec3(d.x, d.y, d.z));
		// invMass(w) = 가벼움 정도. w=0이면 못 움직이는 고정점(마우스로 잡은 점 등)
		denom += (invMass[j] * invMassScale) * (g.x * g.x + g.y * g.y + g.z * g.z);
	}

	// ── Step 5: "그래서 얼마나 세게 밀까?" (XPBD 논문 Eq.18) ──────────────
	//
	//   Δλ = ( −C − α̃·λ ) / ( denom + α̃ )
	//
	//   · −C        : 위반량의 반대. 20% 눌렸으면(C=−0.2) +0.2만큼 되돌리려 한다.
	//   · denom     : 반 전체가 얼마나 잘 반응하는지 (Step 4에서 구함)
	//   · α (compliance) : 물렁한 정도. 단위 [m/N].
	//                 0이면 강철처럼 부피를 절대 안 내준다.
	//                 크면 젤리처럼 "좀 눌려도 괜찮아" 하고 봐준다.
	//   · α̃ = α/dt² : 시간 간격으로 나눠주는 게 XPBD의 핵심.
	//                 이 덕분에 프레임률이나 반복 횟수가 바뀌어도
	//                 '같은 물렁함'이 유지된다. (옛날 PBD는 이게 안 됐다)
	//   · λ (lambda): 지금까지 누적해서 가한 힘. 매 프레임 0에서 시작해서
	//                 반복할수록 쌓인다. 이미 충분히 밀었으면 −α̃·λ 항이
	//                 브레이크를 걸어 과하게 밀지 않게 한다.
	const float dt2 = fmaxf(dt * dt, 1e-8f);
	const float alphaI = alpha_vol ? alpha_vol[idx] : alphaConst;
	const float alphaT = alphaI / dt2;      // α̃
	const float den = denom + alphaT;
	if (!(den > 1e-12f)) return;            // 0으로 나누기 방지

	const float lam = lambda_vol[idx];
	const float dLam = (-C - alphaT * lam) / den;   // 이번 반복에서 추가할 힘
	if (!isfinite(dLam)) return;
	lambda_vol[idx] = lam + dLam;                   // 누적 기록

	// ── Step 6: "구성원 전원에게 '이만큼 움직여' 쪽지를 돌린다" (Eq.17) ──────
	//
	//   Δx_k = w_k · Δλ · g_k        ← 구성원 k가 움직일 양 [m]
	//
	// ★★ 이 커널에서 가장 중요한 부분. 예전에 여기서 '나 하나만' 움직였고
	//    그게 물체가 터지던 원인이었다. 왜 안 되는지:
	//
	//    만약 나는 우리 반의 '한가운데'에 서 있다 → (x_i − c) ≈ 0 → 내 화살표는 거의 0.
	//    반 한가운데 있는 사람 혼자 움직여봐야 반이 차지한 넓이는 안 변한다!
	//    그러면 J가 그대로 → C가 안 줄어듦 → "아직 안 됐네?" 하고 λ만 계속 쌓임
	//    → 매 반복 계속 밀어댐 → 전체가 한 방향으로 표류 → 진동/폭발.
	//
	//    반이 진짜 넓어지려면 '반원들이 바깥으로 흩어져야' 한다.
	//    그래서 전원에게 쪽지를 돌린다. (Region Balloon이 부드러웠던 이유가 이것)
	//
	// atomicAdd를 쓰는 이유: 철수는 내 반의 학생이면서 옆 반, 앞 반의 학생이기도
	//   하다. 여러 반장이 동시에 철수에게 쪽지를 준다. GPU에서 동시에 더하면
	//   값이 깨지므로 atomicAdd로 안전하게 누적한다.
	//   dp_sum[j]에는 쪽지의 합, dp_count[j]에는 받은 쪽지 장수를 센다.
	//   → 나중에 Apply 커널에서 '평균'을 내서 실제로 움직인다.
	//     (합을 그대로 쓰면 반 수십 개가 동시에 밀어 과하게 날아간다)

	// [6-A] 반장인 나 자신도 쪽지를 받는다 (내 화살표는 작지만 0은 아니다)
	{
		const float wi = invMass[idx] * invMassScale;
		if (wi > 0.0f) {                                  // 고정된 점이 아니면
			const float3 dp = f3_mul(g_self, wi * dLam);  // Δx = w·Δλ·g
			if (isfinite(dp.x) && isfinite(dp.y) && isfinite(dp.z)) {
				atomicAdd(&dp_sum[idx].x, dp.x);
				atomicAdd(&dp_sum[idx].y, dp.y);
				atomicAdd(&dp_sum[idx].z, dp.z);
				atomicAdd(&dp_count[idx], 1);             // "쪽지 1장 받았음"
			}
		}
	}

	// [6-B] 반원 전원에게 쪽지 배달 — 여기가 실제로 부피를 바꾸는 부분
	for (int k = 0; k < cnt; ++k) {
		const int j = nbrIdx[off + k];
		if (j < 0 || j >= N) continue;
		const float wj = invMass[j] * invMassScale;
		if (wj <= 0.0f) continue;   // 마우스로 붙잡힌 점/고정 슬랩은 안 밀린다

		// 중심에서 j를 향한 화살표를 다시 구한다 (Step 4에서 저장 안 했으므로)
		const float3 d = f3_sub(pos_pred_in[j], c);
		const glm::vec3 g = gs * (Sinv * glm::vec3(d.x, d.y, d.z));

		// 왼쪽에 선 애는 더 왼쪽으로, 오른쪽에 선 애는 더 오른쪽으로 (J<1일 때)
		const float3 dp = make_float3(g.x * wj * dLam, g.y * wj * dLam, g.z * wj * dLam);
		if (!isfinite(dp.x) || !isfinite(dp.y) || !isfinite(dp.z)) continue;
		atomicAdd(&dp_sum[j].x, dp.x);
		atomicAdd(&dp_sum[j].y, dp.y);
		atomicAdd(&dp_sum[j].z, dp.z);
		atomicAdd(&dp_count[j], 1);
	}
	// → 여기서 커널 종료. 아직 아무도 실제로 움직이지 않았다.
	//   모든 반장이 쪽지를 다 돌리고 나면, Apply 커널이 각자 받은 쪽지를
	//   평균 내서 그제서야 위치를 바꾼다.
}

// ===============================================================
// 패스 2 (Apply): "받은 쪽지를 평균 내서 실제로 움직인다"
//
// 비유: 철수는 자기 반 반장에게도, 옆 반·앞 반 반장에게도 쪽지를 받았다.
//   예를 들어 60장을 받았다면
//     · 어떤 반장은 "왼쪽으로 3cm"
//     · 어떤 반장은 "왼쪽으로 2cm"
//     · 어떤 반장은 "오른쪽으로 1cm"
//   이걸 전부 더해서(=dp_sum) 장수(=dp_count)로 나눈 평균만큼 움직인다.
//
//   왜 합이 아니라 평균인가: 반이 서로 겹쳐 있어서 한 사람이 수십 장을
//   받는다. 다 더해서 움직이면 수십 배로 과하게 날아가 폭발한다.
//   평균을 내면 "여러 반장의 의견을 종합한 한 걸음"이 된다.
//
//   그리고 이웃한 반들은 반원이 거의 겹치므로 쪽지 방향도 거의 같다.
//   → 평균을 내도 서로 상쇄되지 않고 한 방향으로 정렬된다.
//   → 이것이 낱알처럼 흩어지지 않고 '덩어리째 부푸는' 거동의 근원.
// ===============================================================
__global__ void xpbdVolumeApplyKernel(
	int N,
	const float3* pos_pred_in,
	float3* pos_pred_out,
	const float* invMass,
	const float3* dp_sum,
	const int* dp_count,
	const float* restLen,
	float invMassScale,
	float underRelax)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	const float3 pi = pos_pred_in[idx];
	const int cnt = dp_count[idx];                    // 내가 받은 쪽지 장수
	const float wi = invMass[idx] * invMassScale;
	// 쪽지를 한 장도 못 받았거나(주변에 볼륨 반이 없음),
	// 내가 고정점이면(마우스로 잡힘 / squash 슬랩) 제자리에 그대로 둔다.
	if (cnt <= 0 || wi <= 0.0f) { pos_pred_out[idx] = pi; return; }

	// 쪽지들의 평균 = 여러 반장의 의견을 종합한 이번 한 걸음
	float3 avg = f3_mul(dp_sum[idx], 1.0f / (float)cnt);
	if (!isfinite(avg.x) || !isfinite(avg.y) || !isfinite(avg.z)) { pos_pred_out[idx] = pi; return; }

	// 마지막 안전벨트: 한 번에 너무 멀리 못 가게 막는다.
	// restLen = 원래 상태에서 '나 ↔ 반원'의 평균 거리 (내 반의 크기 자)
	// 그 25% 이상은 한 반복에 못 움직인다. 수치 사고가 나도 반을 통째로
	// '뛰어넘어' 다른 곳에 가버리는 참사를 막는 최후 방어선.
	const float rl = restLen[idx];
	if (rl > 1e-12f) {
		const float maxStep = 0.25f * rl;
		const float len = f3_len(avg);
		if (len > maxStep && len > 1e-20f) { atomicAdd(&g_clampCtr[1], 1u); avg = f3_mul(avg, maxStep / len); }
	}

	// underRelax(보통 0.3) = "살살 가자" 계수.
	// 거리 제약·형상 매칭도 동시에 같은 점을 밀고 있으므로, 각자 조금씩만
	// 움직여야 서로 싸우다 진동하지 않는다. 반복을 여러 번 돌면 결국 도달한다.
	pos_pred_out[idx] = f3_add(pi, f3_mul(avg, underRelax));
}

// ===============================================================
// Gather 모드 (scatter와 수학적으로 동일한 계산, 실행 전략만 다름)
//
// scatter: 반장이 반원 전원에게 쪽지를 '배달'한다 (atomicAdd 쓰기, 비결정적)
// gather : 반장은 칠판에 공지만 써 붙이고 (Kernel A),
//          각자가 자기가 속한 반들의 칠판을 '보러 다닌다' (Kernel B, 읽기만)
//
// 장점: atomic 없음(경합/직렬화 소멸), 매 반복 memset 불필요,
//       덧셈 순서가 고정되어 결과가 비트 단위로 재현됨(결정론적).
// 대가: 전치 CSR(+~26MB)과 클러스터 파라미터 버퍼(+4MB)가 필요.
// ===============================================================

// Kernel A: 클러스터당 1스레드. Step 1~5는 scatter 버전과 완전히 동일하고,
// 쪽지 배달 대신 "칠판"(c, Σ⁻¹, 계수)에 결과를 적는다. 쓰기는 자기 슬롯뿐 → atomic 불필요.
__global__ void xpbdVolumeClusterSolveKernel(
	int N,
	const float3* pos_pred_in,
	const float* invMass,
	const int* offset,
	const int* nbrCount,
	const int* nbrIdx,
	const float* detRest,
	const int* matType,
	float* lambda_vol,
	const float* alpha_vol,
	float alphaConst,
	float dt,
	float invMassScale,
	float3* clusterC,     // [N]   out: 무게중심
	float* clusterSinv,   // [N*6] out: Σ⁻¹ (대칭 6원소)
	float* clusterCoef)   // [N]   out: dLam·(J/n). 0 = 이번 반복 무효 클러스터
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	// 실패 경로가 어디서 return 하든 "이 클러스터는 무효"가 칠판에 남도록 먼저 0을 쓴다.
	clusterCoef[idx] = 0.0f;

	const float3 pi = pos_pred_in[idx];
	if (matType[idx] != 0) return;

	const int off = offset[idx];
	const int cnt = nbrCount[idx];

	// ── Step 1: 무게중심 & 공분산 (scatter 버전과 동일) ──
	float3 c = pi;
	int used = 0;
	for (int k = 0; k < cnt; ++k) {
		const int j = nbrIdx[off + k];
		if (j < 0 || j >= N) continue;
		c = f3_add(c, pos_pred_in[j]);
		++used;
	}
	if (used < 3) return;
	const float n = (float)(used + 1);
	const float invN = 1.0f / n;
	c = f3_mul(c, invN);

	glm::mat3 Scur(0.0f);
	{
		const float3 d = f3_sub(pi, c);
		const glm::vec3 v(d.x, d.y, d.z);
		Scur += glm::outerProduct(v, v);
	}
	for (int k = 0; k < cnt; ++k) {
		const int j = nbrIdx[off + k];
		if (j < 0 || j >= N) continue;
		const float3 d = f3_sub(pos_pred_in[j], c);
		const glm::vec3 v(d.x, d.y, d.z);
		Scur += glm::outerProduct(v, v);
	}
	Scur *= invN;

	// ── Step 2: J, C (동일) ──
	float sCur, detHatCur;
	if (!sigmaScaleAndShapeDet(Scur, sCur, detHatCur)) return;
	const float detCur = sCur * sCur * sCur * detHatCur;
	const float dRest = detRest[idx];
	if (dRest < 1e-30f) return;
	float J = sqrtf(detCur / dRest);
	if (!isfinite(J) || J <= 0.0f) return;
	// [안전판] J 하드 클램프. (범인이 아니라 극단값 방어용 안전망. 진짜 폭주 원인은 restLen이었음)
	// 물리적으로 한 프레임에 부피 3배는 불가능하므로 J를 제한한다.
	if (J > 3.0f || J < 0.333f) { atomicAdd(&g_clampCtr[0], 1u); J = fminf(3.0f, fmaxf(0.333f, J)); }
	const float C = J - 1.0f;
	if (fabsf(C) < 1e-4f) return;

	// ── Step 3: Σ⁻¹ (축퇴 방어 — 두 모드, 위 g_dSinvIsoReg 주석 참조) ──
	glm::mat3 Sinv;
	if (g_dSinvIsoReg) {
		// [모드 1] 등방 정규화. sCur는 Step 2에서 이미 구한 trace/3 이라 공짜.
		glm::mat3 Sreg = Scur;
		if (detHatCur < g_dSinvIsoDetThr) {          // 축퇴한 클러스터만
			atomicAdd(&g_clampCtr[2], 1u);
			const float d = g_dSinvIsoEps * sCur;
			Sreg[0][0] += d; Sreg[1][1] += d; Sreg[2][2] += d;
		}
		if (!inverse3x3_safe(Sreg, Sinv, 1e-20f)) return;
	}
	else {
		// [모드 0] 고유분해 + 최소축 클램프 (기존 G4)
		glm::vec3 eig;
		glm::mat3 V;
		eigenDecomposition_glm(Scur, eig, V);
		const float lmax = fmaxf(eig.z, 1e-20f);
		const float lmin = 0.06f * lmax;
		if (eig.x < lmin || eig.y < lmin) atomicAdd(&g_clampCtr[2], 1u); // G4 실제 발동
		const glm::vec3 e = glm::max(eig, glm::vec3(lmin));
		glm::mat3 D(0.0f);
		D[0][0] = e.x; D[1][1] = e.y; D[2][2] = e.z;
		const glm::mat3 Sclamped = V * D * glm::transpose(V);
		if (!inverse3x3_safe(Sclamped, Sinv, 1e-20f)) return;
	}

	// ── Step 4: denom (동일 — 전 멤버가 움직인다는 가정) ──
	const float gs = J * invN;
	float denom = 0.0f;
	{
		const float3 d = f3_sub(pi, c);
		const glm::vec3 g = gs * (Sinv * glm::vec3(d.x, d.y, d.z));
		denom += (invMass[idx] * invMassScale) * (g.x * g.x + g.y * g.y + g.z * g.z);
	}
	for (int k = 0; k < cnt; ++k) {
		const int j = nbrIdx[off + k];
		if (j < 0 || j >= N) continue;
		const float3 d = f3_sub(pos_pred_in[j], c);
		const glm::vec3 g = gs * (Sinv * glm::vec3(d.x, d.y, d.z));
		denom += (invMass[j] * invMassScale) * (g.x * g.x + g.y * g.y + g.z * g.z);
	}

	// ── Step 5: Δλ (동일) ──
	const float dt2 = fmaxf(dt * dt, 1e-8f);
	const float alphaI = alpha_vol ? alpha_vol[idx] : alphaConst;
	const float alphaT = alphaI / dt2;
	const float den = denom + alphaT;
	if (!(den > 1e-12f)) return;
	const float lam = lambda_vol[idx];
	const float dLam = (-C - alphaT * lam) / den;
	if (!isfinite(dLam)) return;
	lambda_vol[idx] = lam + dLam;

	// ── Step 6 대신: 칠판에 적는다 ──
	// 멤버 j의 이동량 Δx_j = w_j · dLam · gs · Σ⁻¹ (x_j − c) 이므로,
	// (dLam·gs)를 계수 하나로 합쳐 저장하면 Kernel B는 행렬-벡터 곱 한 번이면 된다.
	clusterC[idx] = c;
	storeSym3(clusterSinv + idx * 6, Sinv);
	clusterCoef[idx] = dLam * gs;
}

// ───────────────────────────────────────────────────────────────────────────
// Kernel A' : 클러스터당 1'워프'(32스레드). Kernel A와 수학적으로 완전히 동일하다.
//
// 왜 만드나 — Kernel A는 스레드 하나가 멤버 n명을 혼자 훑는데, n개 위치를
// 보관할 수 없어 무게중심·공분산·denom 세 패스에서 전역 메모리를 다시 읽는다.
// 이 커널은 산술보다 메모리 대역폭에 묶이므로(~1 FLOP/byte) 재읽기가 비용을 지배한다.
//
// 워프로 나누면 lane 하나가 n/32명만 맡고, 앞 VOL_WARP_CACHE개를 캐시해 재사용한다.
// 다만 이 배열은 동적 인덱싱 때문에 컴파일러가 로컬 스택에 spill할 수 있으며, 캐시를
// 넘는 멤버는 여전히 전역에서 재읽는다. 아래 A'' 1-pass가 이 한계를 없애는 비교 대상이다.
// (부수적으로 클러스터별 n 편차로 인한 워프 발산도 사라진다)
//
// 멤버 분배: lane L 이 k = L, L+32, L+64 ... 를 맡는다. 32개 lane이 동시에 읽는
// nbrIdx[off+lane] 이 연속 주소가 되어 coalesced 로 들어온다.
//
// 합산: __shfl_xor_sync 버터플라이 전-리덕션(5단계). shared memory도
// __syncthreads()도 필요 없고 결과가 32개 lane 전부에 남는다.
// 합칠 값은 11개 — 무게중심 3 + 유효 멤버 수 1 + 공분산 6 + denom 1.
//
// J·Σ⁻¹(O(1) 구간)은 32개 lane이 '중복으로 똑같이' 계산한다. 한 lane이 계산하고
// 방송하는 것보다 싸고(어차피 놀 스레드), 분기가 워프 균일이라 발산이 없다.
//
// ⚠️ n이 작으면 리덕션 오버헤드(shuffle 55회)가 실제 계산을 잡아먹는다.
//   n=64  → lane당 2명, 계산 ~106 FLOP  vs 오버헤드 55  → 이득 거의 없음
//   n=257 → lane당 8명, 계산 ~424 FLOP  vs 오버헤드 55  → 이득 큼
//   따라서 cap을 키우거나 없애는 것과 세트로 써야 한다.
// ───────────────────────────────────────────────────────────────────────────
#define VOL_WARP_CACHE 8   // lane당 위치/인덱스 캐시 슬롯 (cap 256까지 전원 커버; 실제 배치는 컴파일러가 결정)

// 워프 전-리덕션: 결과가 32개 lane 전부에 남는다.
__device__ __forceinline__ float warpAllSum(float v)
{
	for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);//butterfly sum
	return v;
}

__global__ void xpbdVolumeClusterSolveWarpKernel(
	int N,
	const float3* pos_pred_in,
	const float* invMass,
	const int* offset,
	const int* nbrCount,
	const int* nbrIdx,
	const float* detRest,
	const int* matType,
	float* lambda_vol,
	const float* alpha_vol,
	float alphaConst,
	float dt,
	float invMassScale,
	float3* clusterC,
	float* clusterSinv,
	float* clusterCoef)
{
	const int warpsPerBlock = blockDim.x / 32;// 256/32 8
	const int lane = threadIdx.x % 32;// warp 안에서 32개중 하나
	const int idx = blockIdx.x * warpsPerBlock + (threadIdx.x / 32); // 클러스터 = 노드 idx
	// 아래 return들은 전부 idx에만 의존하므로 워프 균일하다(발산 없음).
	if (idx >= N) return;
	if (matType[idx] != 0) { if (lane == 0) clusterCoef[idx] = 0.0f; return; }

	const int off = offset[idx];
	const int cnt = nbrCount[idx];
	const float3 pi = pos_pred_in[idx];// 반장

	// ── Pass 1: 무게중심 + 위치/인덱스를 레지스터에 캐시 ──
	float3 psum = make_float3(0.0f, 0.0f, 0.0f);
	int ucnt = 0, nCache = 0;
	float3 cPos[VOL_WARP_CACHE];
	int    cJ[VOL_WARP_CACHE];
	for (int k = lane; k < cnt; k += 32) {// 각 LANE 마다 32개씩건너뛰며 이웃들을 담당
		const int j = nbrIdx[off + k];
		if (j < 0 || j >= N) continue;
		const float3 p = pos_pred_in[j];
		psum = f3_add(psum, p);
		++ucnt;
		if (nCache < VOL_WARP_CACHE) { cPos[nCache] = p; cJ[nCache] = j; ++nCache; }
	}
	const float sx = warpAllSum(psum.x);
	const float sy = warpAllSum(psum.y);
	const float sz = warpAllSum(psum.z);
	const int used = (int)(warpAllSum((float)ucnt) + 0.5f);
	if (used < 3) { if (lane == 0) clusterCoef[idx] = 0.0f; return; }

	const float n = (float)(used + 1);
	const float invN = 1.0f / n;
	const float3 c = make_float3((sx + pi.x) * invN, (sy + pi.y) * invN, (sz + pi.z) * invN);//무게중심

	// ── Pass 2: 공분산 (캐시된 멤버는 레지스터에서, 넘친 만큼만 전역 재읽기) ──
	float a00 = 0.f, a11 = 0.f, a22 = 0.f, a01 = 0.f, a02 = 0.f, a12 = 0.f;
	for (int t = 0; t < nCache; ++t) {//레지스터 캐시된 이웃들 각자 lane 별 담당한 이웃cPos 에담긴것들 순회하며 계산
		const float dx = cPos[t].x - c.x, dy = cPos[t].y - c.y, dz = cPos[t].z - c.z;
		a00 += dx * dx; a11 += dy * dy; a22 += dz * dz;
		a01 += dx * dy; a02 += dx * dz; a12 += dy * dz;
	}
	if (ucnt > nCache) { // lane당 멤버가 VOL_WARP_CACHE를 넘는 경우만 여기가 병목임.
		int seen = 0;
		for (int k = lane; k < cnt; k += 32) {
			const int j = nbrIdx[off + k];
			if (j < 0 || j >= N) continue;
			if (seen++ < nCache) continue;
			const float3 p = pos_pred_in[j];
			const float dx = p.x - c.x, dy = p.y - c.y, dz = p.z - c.z;
			a00 += dx * dx; a11 += dy * dy; a22 += dz * dz;
			a01 += dx * dy; a02 += dx * dz; a12 += dy * dz;
		}
	}
	if (lane == 0) { // 반장 자신의 기여는 한 lane만
		const float dx = pi.x - c.x, dy = pi.y - c.y, dz = pi.z - c.z;
		a00 += dx * dx; a11 += dy * dy; a22 += dz * dz;
		a01 += dx * dy; a02 += dx * dz; a12 += dy * dz;
	}
	glm::mat3 Scur(0.0f);//공분산 행렬
	Scur[0][0] = warpAllSum(a00) * invN; 
	Scur[1][1] = warpAllSum(a11) * invN;
	Scur[2][2] = warpAllSum(a22) * invN;
	const float 
		s01 = warpAllSum(a01) * invN, 
		s02 = warpAllSum(a02) * invN,
		s12 = warpAllSum(a12) * invN;
	Scur[0][1] = Scur[1][0] = s01; 
	Scur[0][2] = Scur[2][0] = s02;
	Scur[1][2] = Scur[2][1] = s12;

	// ── Step 2~3: J, Σ⁻¹ — 32개 lane이 같은 입력으로 중복 계산 (분기 균일) ──
	float sCur, detHatCur;
	if (!sigmaScaleAndShapeDet(Scur, sCur, detHatCur)) { if (lane == 0) clusterCoef[idx] = 0.0f; return; }
	const float detCur = sCur * sCur * sCur * detHatCur;
	const float dRest = detRest[idx];
	if (dRest < 1e-30f) { if (lane == 0) clusterCoef[idx] = 0.0f; return; }
	float J = sqrtf(detCur / dRest);
	if (!isfinite(J) || J <= 0.0f) { if (lane == 0) clusterCoef[idx] = 0.0f; return; }
	if (J > 3.0f || J < 0.333f) { if (lane == 0) atomicAdd(&g_clampCtr[0], 1u); J = fminf(3.0f, fmaxf(0.333f, J)); }
	const float C = J - 1.0f;
	if (fabsf(C) < 1e-4f) { if (lane == 0) clusterCoef[idx] = 0.0f; return; }

	glm::mat3 Sinv;
	if (g_dSinvIsoReg) {
		glm::mat3 Sreg = Scur;
		if (detHatCur < g_dSinvIsoDetThr) {
			if (lane == 0) atomicAdd(&g_clampCtr[2], 1u);
			const float d = g_dSinvIsoEps * sCur;
			Sreg[0][0] += d; Sreg[1][1] += d; Sreg[2][2] += d;
		}
		if (!inverse3x3_safe(Sreg, Sinv, 1e-20f)) { if (lane == 0) clusterCoef[idx] = 0.0f; return; }
	}
	else {
		glm::vec3 eig; glm::mat3 V;
		eigenDecomposition_glm(Scur, eig, V);
		const float lmax = fmaxf(eig.z, 1e-20f);
		const float lmin = 0.06f * lmax;
		if ((eig.x < lmin || eig.y < lmin) && lane == 0) atomicAdd(&g_clampCtr[2], 1u);
		const glm::vec3 e = glm::max(eig, glm::vec3(lmin));
		glm::mat3 D(0.0f); D[0][0] = e.x; D[1][1] = e.y; D[2][2] = e.z;
		const glm::mat3 Sclamped = V * D * glm::transpose(V);
		if (!inverse3x3_safe(Sclamped, Sinv, 1e-20f)) { if (lane == 0) clusterCoef[idx] = 0.0f; return; }
	}

	// ── Pass 4: denom (다시 레지스터 재사용) ──
	const float gs = J * invN;
	float dsum = 0.0f;
	for (int t = 0; t < nCache; ++t) {
		const glm::vec3 g = gs * (Sinv * glm::vec3(cPos[t].x - c.x, cPos[t].y - c.y, cPos[t].z - c.z));
		dsum += (invMass[cJ[t]] * invMassScale) * (g.x * g.x + g.y * g.y + g.z * g.z);
	}
	if (ucnt > nCache) {
		int seen = 0;
		for (int k = lane; k < cnt; k += 32) {
			const int j = nbrIdx[off + k];
			if (j < 0 || j >= N) continue;
			if (seen++ < nCache) continue;
			const float3 p = pos_pred_in[j];
			const glm::vec3 g = gs * (Sinv * glm::vec3(p.x - c.x, p.y - c.y, p.z - c.z));
			dsum += (invMass[j] * invMassScale) * (g.x * g.x + g.y * g.y + g.z * g.z);
		}
	}
	if (lane == 0) {
		const glm::vec3 g = gs * (Sinv * glm::vec3(pi.x - c.x, pi.y - c.y, pi.z - c.z));
		dsum += (invMass[idx] * invMassScale) * (g.x * g.x + g.y * g.y + g.z * g.z);
	}
	const float denom = warpAllSum(dsum);

	// ── Step 5~6: Δλ 와 칠판 — lane 0만 쓴다 ──
	if (lane != 0) return;
	const float dt2 = fmaxf(dt * dt, 1e-8f);
	const float alphaI = alpha_vol ? alpha_vol[idx] : alphaConst;
	const float alphaT = alphaI / dt2;
	const float den = denom + alphaT;
	if (!(den > 1e-12f)) { clusterCoef[idx] = 0.0f; return; }
	const float lam = lambda_vol[idx];
	const float dLam = (-C - alphaT * lam) / den;
	if (!isfinite(dLam)) { clusterCoef[idx] = 0.0f; return; }
	lambda_vol[idx] = lam + dLam;

	clusterC[idx] = c;
	storeSym3(clusterSinv + idx * 6, Sinv);
	clusterCoef[idx] = dLam * gs;
}

// Kernel A'' : 클러스터당 1워프, 1-pass 모멘트 방식.
//
// 캐시형 A'는 각 lane이 cPos[8]/cJ[8]에 멤버를 보관한 뒤 공분산과 denom에서 다시
// 사용한다. 여기서는 멤버 위치를 보관하지 않는다. 반장 pi 기준 상대좌표 e=p-pi를
// 한 번만 읽어 다음 통계량을 누적하고 즉시 버린다.
//
//   q = sum(e)/n,  c = pi+q,  Sigma = sum(ee^T)/n - qq^T
//   M = sum(w (p-c)(p-c)^T)
//     = Q - u q^T - q u^T + W qq^T
//   denom = (J/n)^2 tr(Sinv^T Sinv M)
//
// 따라서 기존 Pass 4의 sum w |(J/n) Sinv(p-c)|^2와 수학적으로 같다.
// pi를 기준으로 누적하므로 큰 월드 좌표를 직접 빼는 수치 상쇄도 피한다.
__global__ void xpbdVolumeClusterSolveWarpOnePassKernel(
	int N,
	const float3* pos_pred_in,
	const float* invMass,
	const int* offset,
	const int* nbrCount,
	const int* nbrIdx,
	const float* detRest,
	const int* matType,
	float* lambda_vol,
	const float* alpha_vol,
	float alphaConst,
	float dt,
	float invMassScale,
	float3* clusterC,
	float* clusterSinv,
	float* clusterCoef)
{
	const int warpsPerBlock = blockDim.x / 32;
	const int lane = threadIdx.x % 32;
	const int idx = blockIdx.x * warpsPerBlock + (threadIdx.x / 32);
	if (idx >= N) return;
	if (matType[idx] != 0) { if (lane == 0) clusterCoef[idx] = 0.0f; return; }

	const int off = offset[idx];
	const int cnt = nbrCount[idx];
	const float3 pi = pos_pred_in[idx];

	// Pass 1 only: unweighted moments build Sigma; weighted moments later rebuild denom.
	float sx = 0.0f, sy = 0.0f, sz = 0.0f;// 기본 무게중심점 대신 반장 p를 기준으로 이웃들 상대벡터합
	float e00 = 0.0f, e11 = 0.0f, e22 = 0.0f, e01 = 0.0f, e02 = 0.0f, e12 = 0.0f;//공분산 행렬 구할시 필요한 ekek^T 이웃들 합
	float W = 0.0f, ux = 0.0f, uy = 0.0f, uz = 0.0f;
	float q00 = 0.0f, q11 = 0.0f, q22 = 0.0f, q01 = 0.0f, q02 = 0.0f, q12 = 0.0f;
	int ucnt = 0;

	for (int k = lane; k < cnt; k += 32) {
		const int j = nbrIdx[off + k];
		if (j < 0 || j >= N) continue;

		const float3 p = pos_pred_in[j];
		const float ex = p.x - pi.x, ey = p.y - pi.y, ez = p.z - pi.z;//반장을 중심으로 상대위치계산
		sx += ex; sy += ey; sz += ez;//합
		e00 += ex * ex; e11 += ey * ey; e22 += ez * ez;
		e01 += ex * ey; e02 += ex * ez; e12 += ey * ez;

		const float w = invMass[j] * invMassScale;//역질량
		W += w;
		ux += w * ex; uy += w * ey; uz += w * ez;
		q00 += w * ex * ex; q11 += w * ey * ey; q22 += w * ez * ez;
		q01 += w * ex * ey; q02 += w * ex * ez; q12 += w * ey * ez;
		++ucnt;
	}

	const int used = (int)(warpAllSum((float)ucnt) + 0.5f);
	if (used < 3) { if (lane == 0) clusterCoef[idx] = 0.0f; return; }

	const float n = (float)(used + 1);
	const float invN = 1.0f / n;
	const float qx = warpAllSum(sx) * invN;
	const float qy = warpAllSum(sy) * invN;
	const float qz = warpAllSum(sz) * invN;
	const float3 c = make_float3(pi.x + qx, pi.y + qy, pi.z + qz);//무게중심

	// Sigma = E/n - qq^T. The leader has e=0 and is already included through n.
	const float s00 = warpAllSum(e00) * invN - qx * qx;
	const float s11 = warpAllSum(e11) * invN - qy * qy;
	const float s22 = warpAllSum(e22) * invN - qz * qz;
	const float s01 = warpAllSum(e01) * invN - qx * qy;
	const float s02 = warpAllSum(e02) * invN - qx * qz;
	const float s12 = warpAllSum(e12) * invN - qy * qz;

	glm::mat3 Scur(0.0f);
	Scur[0][0] = s00; Scur[1][1] = s11; Scur[2][2] = s22;
	Scur[0][1] = Scur[1][0] = s01;
	Scur[0][2] = Scur[2][0] = s02;
	Scur[1][2] = Scur[2][1] = s12;

	float sCur, detHatCur;
	if (!sigmaScaleAndShapeDet(Scur, sCur, detHatCur)) { if (lane == 0) clusterCoef[idx] = 0.0f; return; }
	const float detCur = sCur * sCur * sCur * detHatCur;
	const float dRest = detRest[idx];
	if (dRest < 1e-30f) { if (lane == 0) clusterCoef[idx] = 0.0f; return; }
	float J = sqrtf(detCur / dRest);
	if (!isfinite(J) || J <= 0.0f) { if (lane == 0) clusterCoef[idx] = 0.0f; return; }
	if (J > 3.0f || J < 0.333f) { if (lane == 0) atomicAdd(&g_clampCtr[0], 1u); J = fminf(3.0f, fmaxf(0.333f, J)); }
	const float C = J - 1.0f;
	if (fabsf(C) < 1e-4f) { if (lane == 0) clusterCoef[idx] = 0.0f; return; }

	glm::mat3 Sinv;
	if (g_dSinvIsoReg) {
		glm::mat3 Sreg = Scur;
		if (detHatCur < g_dSinvIsoDetThr) {
			if (lane == 0) atomicAdd(&g_clampCtr[2], 1u);
			const float d = g_dSinvIsoEps * sCur;
			Sreg[0][0] += d; Sreg[1][1] += d; Sreg[2][2] += d;
		}
		if (!inverse3x3_safe(Sreg, Sinv, 1e-20f)) { if (lane == 0) clusterCoef[idx] = 0.0f; return; }
	}
	else {
		glm::vec3 eig; glm::mat3 V;
		eigenDecomposition_glm(Scur, eig, V);
		const float lmax = fmaxf(eig.z, 1e-20f);
		const float lmin = 0.06f * lmax;
		if ((eig.x < lmin || eig.y < lmin) && lane == 0) atomicAdd(&g_clampCtr[2], 1u);
		const glm::vec3 e = glm::max(eig, glm::vec3(lmin));
		glm::mat3 D(0.0f); D[0][0] = e.x; D[1][1] = e.y; D[2][2] = e.z;
		const glm::mat3 Sclamped = V * D * glm::transpose(V);
		if (!inverse3x3_safe(Sclamped, Sinv, 1e-20f)) { if (lane == 0) clusterCoef[idx] = 0.0f; return; }
	}

	// M is the exact weighted centered second moment used by the original denom pass.
	// The leader's e=0 contribution is added by putting its mass into W before reduction.
	if (lane == 0) W += invMass[idx] * invMassScale;
	W = warpAllSum(W);
	ux = warpAllSum(ux); uy = warpAllSum(uy); uz = warpAllSum(uz);
	q00 = warpAllSum(q00); q11 = warpAllSum(q11); q22 = warpAllSum(q22);
	q01 = warpAllSum(q01); q02 = warpAllSum(q02); q12 = warpAllSum(q12);

	const float m00 = q00 - 2.0f * ux * qx + W * qx * qx;
	const float m11 = q11 - 2.0f * uy * qy + W * qy * qy;
	const float m22 = q22 - 2.0f * uz * qz + W * qz * qz;
	const float m01 = q01 - ux * qy - qx * uy + W * qx * qy;
	const float m02 = q02 - ux * qz - qx * uz + W * qx * qz;
	const float m12 = q12 - uy * qz - qy * uz + W * qy * qz;

	// H=Sinv^T Sinv. trace(HM) is sum w |Sinv(p-c)|^2 without another member pass.
	const float r00 = Sinv[0][0], r01 = Sinv[1][0], r02 = Sinv[2][0];
	const float r10 = Sinv[0][1], r11 = Sinv[1][1], r12 = Sinv[2][1];
	const float r20 = Sinv[0][2], r21 = Sinv[1][2], r22 = Sinv[2][2];
	const float h00 = r00 * r00 + r10 * r10 + r20 * r20;
	const float h11 = r01 * r01 + r11 * r11 + r21 * r21;
	const float h22 = r02 * r02 + r12 * r12 + r22 * r22;
	const float h01 = r00 * r01 + r10 * r11 + r20 * r21;
	const float h02 = r00 * r02 + r10 * r12 + r20 * r22;
	const float h12 = r01 * r02 + r11 * r12 + r21 * r22;
	const float momentTrace = h00 * m00 + h11 * m11 + h22 * m22 +
		2.0f * (h01 * m01 + h02 * m02 + h12 * m12);
	if (!isfinite(momentTrace)) { if (lane == 0) clusterCoef[idx] = 0.0f; return; }
	const float gs = J * invN;
	const float denom = fmaxf(0.0f, gs * gs * momentTrace);

	if (lane != 0) return;
	const float dt2 = fmaxf(dt * dt, 1e-8f);
	const float alphaI = alpha_vol ? alpha_vol[idx] : alphaConst;
	const float alphaT = alphaI / dt2;
	const float den = denom + alphaT;
	if (!(den > 1e-12f)) { clusterCoef[idx] = 0.0f; return; }
	const float lam = lambda_vol[idx];
	const float dLam = (-C - alphaT * lam) / den;
	if (!isfinite(dLam)) { clusterCoef[idx] = 0.0f; return; }
	lambda_vol[idx] = lam + dLam;

	clusterC[idx] = c;
	storeSym3(clusterSinv + idx * 6, Sinv);
	clusterCoef[idx] = dLam * gs;
}

// 부피 제약 Gauss-Seidel: 한 색(멤버를 하나도 공유하지 않는 클러스터 묶음)을 클러스터당 1스레드로 푼다.
// Step 1~5 는 xpbdVolumeClusterSolveKernel 과 한 줄씩 같다 (기존 경로는 건드리지 않으려고 복사했다).
// 차이는 Step 6 뿐 — 칠판에 적고 평균하는 대신 전 멤버(자신 포함)를 바로 옮긴다.
//   · 같은 색 안의 클러스터는 멤버가 겹치지 않으므로 제자리 쓰기에 경쟁이 없다.
//   · 다음 색은 갱신된 위치를 보므로 겹침 평균(V2)이 필요 없다 = 각 클러스터가 제약을 온전히 푼다.
//   · 스텝 클램프(V4, 0.25·restLen)는 멤버마다 그대로 둔다. underRelax 는 적용하지 않는다(GS 는 ω=1).
__global__ void xpbdVolumeGSColorKernel(
	int begin,
	int count,
	const int* order,        // 색 순서로 정렬된 클러스터(리더) 인덱스
	int N,
	float3* pos,             // 제자리 갱신
	const float* invMass,
	const int* offset,
	const int* nbrCount,
	const int* nbrIdx,
	const float* detRest,
	const int* matType,
	float* lambda_vol,
	const float* alpha_vol,
	float alphaConst,
	float dt,
	float invMassScale,
	const float* restLen)
{
	const int t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= count) return;
	const int idx = order[begin + t];
	if (idx < 0 || idx >= N) return;
	if (matType[idx] != 0) return;

	const float3 pi = pos[idx];
	const int off = offset[idx];
	const int cnt = nbrCount[idx];

	// ── Step 1: 무게중심 & 공분산 ──
	float3 c = pi;
	int used = 0;
	for (int k = 0; k < cnt; ++k) {
		const int j = nbrIdx[off + k];
		if (j < 0 || j >= N) continue;
		c = f3_add(c, pos[j]);
		++used;
	}
	if (used < 3) return;
	const float n = (float)(used + 1);
	const float invN = 1.0f / n;
	c = f3_mul(c, invN);

	glm::mat3 Scur(0.0f);
	{
		const float3 d = f3_sub(pi, c);
		const glm::vec3 v(d.x, d.y, d.z);
		Scur += glm::outerProduct(v, v);
	}
	for (int k = 0; k < cnt; ++k) {
		const int j = nbrIdx[off + k];
		if (j < 0 || j >= N) continue;
		const float3 d = f3_sub(pos[j], c);
		const glm::vec3 v(d.x, d.y, d.z);
		Scur += glm::outerProduct(v, v);
	}
	Scur *= invN;

	// ── Step 2: J, C ──
	float sCur, detHatCur;
	if (!sigmaScaleAndShapeDet(Scur, sCur, detHatCur)) return;
	const float detCur = sCur * sCur * sCur * detHatCur;
	const float dRest = detRest[idx];
	if (dRest < 1e-30f) return;
	float J = sqrtf(detCur / dRest);
	if (!isfinite(J) || J <= 0.0f) return;
	if (J > 3.0f || J < 0.333f) { atomicAdd(&g_clampCtr[0], 1u); J = fminf(3.0f, fmaxf(0.333f, J)); }
	const float C = J - 1.0f;
	if (fabsf(C) < 1e-4f) return;

	// ── Step 3: Σ⁻¹ (축퇴 방어) ──
	glm::mat3 Sinv;
	if (g_dSinvIsoReg) {
		glm::mat3 Sreg = Scur;
		if (detHatCur < g_dSinvIsoDetThr) {
			atomicAdd(&g_clampCtr[2], 1u);
			const float d = g_dSinvIsoEps * sCur;
			Sreg[0][0] += d; Sreg[1][1] += d; Sreg[2][2] += d;
		}
		if (!inverse3x3_safe(Sreg, Sinv, 1e-20f)) return;
	}
	else {
		glm::vec3 eig;
		glm::mat3 V;
		eigenDecomposition_glm(Scur, eig, V);
		const float lmax = fmaxf(eig.z, 1e-20f);
		const float lmin = 0.06f * lmax;
		if (eig.x < lmin || eig.y < lmin) atomicAdd(&g_clampCtr[2], 1u);
		const glm::vec3 e = glm::max(eig, glm::vec3(lmin));
		glm::mat3 D(0.0f);
		D[0][0] = e.x; D[1][1] = e.y; D[2][2] = e.z;
		const glm::mat3 Sclamped = V * D * glm::transpose(V);
		if (!inverse3x3_safe(Sclamped, Sinv, 1e-20f)) return;
	}

	// ── Step 4: denom ──
	const float gs = J * invN;
	float denom = 0.0f;
	{
		const float3 d = f3_sub(pi, c);
		const glm::vec3 g = gs * (Sinv * glm::vec3(d.x, d.y, d.z));
		denom += (invMass[idx] * invMassScale) * (g.x * g.x + g.y * g.y + g.z * g.z);
	}
	for (int k = 0; k < cnt; ++k) {
		const int j = nbrIdx[off + k];
		if (j < 0 || j >= N) continue;
		const float3 d = f3_sub(pos[j], c);
		const glm::vec3 g = gs * (Sinv * glm::vec3(d.x, d.y, d.z));
		denom += (invMass[j] * invMassScale) * (g.x * g.x + g.y * g.y + g.z * g.z);
	}

	// ── Step 5: Δλ ──
	const float dt2 = fmaxf(dt * dt, 1e-8f);
	const float alphaI = alpha_vol ? alpha_vol[idx] : alphaConst;
	const float alphaT = alphaI / dt2;
	const float den = denom + alphaT;
	if (!(den > 1e-12f)) return;
	const float lam = lambda_vol[idx];
	const float dLam = (-C - alphaT * lam) / den;
	if (!isfinite(dLam)) return;
	lambda_vol[idx] = lam + dLam;

	// ── Step 6: 전 멤버 제자리 적용 (Δx_j = w_j · dLam · gs · Σ⁻¹ (x_j − c)) ──
	const float coef = dLam * gs;
	for (int k = -1; k < cnt; ++k) {
		const int j = (k < 0) ? idx : nbrIdx[off + k];
		if (j < 0 || j >= N) continue;
		const float wj = invMass[j] * invMassScale;
		if (wj <= 0.0f) continue;
		const float3 xj = pos[j];
		const float3 d = f3_sub(xj, c);
		const glm::vec3 g = Sinv * glm::vec3(d.x, d.y, d.z);
		float3 dp = make_float3(g.x * wj * coef, g.y * wj * coef, g.z * wj * coef);
		if (!isfinite(dp.x) || !isfinite(dp.y) || !isfinite(dp.z)) continue;
		const float rl = restLen[j];
		if (rl > 1e-12f) {
			const float maxStep = 0.25f * rl;
			const float len = f3_len(dp);
			if (len > maxStep && len > 1e-20f) { atomicAdd(&g_clampCtr[1], 1u); dp = f3_mul(dp, maxStep / len); }
		}
		pos[j] = f3_add(xj, dp);
	}
}

// Kernel B: 입자당 1스레드. 자기를 멤버로 포함하는 클러스터들의 칠판을 읽어서
// Δx = w·coef_i·Σ_i⁻¹(x − c_i) 를 합산하고, 평균·클램프·적용한다. 쓰기는 자기 위치뿐.
__global__ void xpbdVolumeGatherApplyKernel(
	int N,
	const float3* pos_pred_in,
	float3* pos_pred_out,
	const float* invMass,
	const int* revOffset,
	const int* revCount,
	const int* revIdx,
	const float3* clusterC,
	const float* clusterSinv,
	const float* clusterCoef,
	const float* restLen,
	float invMassScale,
	float underRelax)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	const float3 pi = pos_pred_in[idx];
	const float wi = invMass[idx] * invMassScale;
	if (wi <= 0.0f) { pos_pred_out[idx] = pi; return; } // 고정점은 어떤 반의 지시도 안 받는다

	const int off = revOffset[idx];
	const int cnt = revCount[idx];

	float3 sum = make_float3(0.0f, 0.0f, 0.0f);
	int received = 0;
	for (int k = 0; k < cnt; ++k) {
		const int i = revIdx[off + k];          // 나를 포함하는 클러스터 i
		if (i < 0 || i >= N) continue;
		const float coef = clusterCoef[i];
		if (coef == 0.0f) continue;             // 이번 반복에 무효인 클러스터 (게이트/데드밴드/실패)

		const glm::mat3 Sinv = loadSym3(clusterSinv + i * 6);
		const float3 d = f3_sub(pi, clusterC[i]);
		const glm::vec3 g = Sinv * glm::vec3(d.x, d.y, d.z);
		const float3 dp = make_float3(g.x * wi * coef, g.y * wi * coef, g.z * wi * coef);
		if (!isfinite(dp.x) || !isfinite(dp.y) || !isfinite(dp.z)) continue;
		sum = f3_add(sum, dp);
		++received;
	}

	if (received <= 0) { pos_pred_out[idx] = pi; return; }

	// scatter의 Apply와 동일: 평균 → 스텝 클램프 → underRelax
	float3 avg = f3_mul(sum, 1.0f / (float)received);
	if (!isfinite(avg.x) || !isfinite(avg.y) || !isfinite(avg.z)) { pos_pred_out[idx] = pi; return; }

	const float rl = restLen[idx];
	if (rl > 1e-12f) {
		const float maxStep = 0.25f * rl;
		const float len = f3_len(avg);
		if (len > maxStep && len > 1e-20f) { atomicAdd(&g_clampCtr[1], 1u); avg = f3_mul(avg, maxStep / len); }
	}

	pos_pred_out[idx] = f3_add(pi, f3_mul(avg, underRelax));
}

// ───────────────────────────────────────────────────────────────────────────
// Kernel B' : 입자당 1'워프'. Kernel B와 수학적으로 동일하다.
//
// 기존 B의 문제는 트래픽이 아니라 부하 불균형이다. 칠판은 원래 한 번씩만 읽으므로
// Kernel A'처럼 '3패스 → 1패스' 로 줄일 여지가 없다. 대신 —
//   워프 32스레드가 서로 다른 가우시안 32개를 맡는데 h_j(소속 클러스터 수)가
//   0~937로 제각각이다. 워프는 최장 스레드를 기다리므로 나머지가 놀고,
//   32개 표본의 최댓값은 평균의 2~3배가 되기 쉽다 → 효율 30~50%.
// 워프 하나가 가우시안 하나를 맡으면 그 편차가 워프 '사이'로 옮겨가고,
// 블록/워프 스케줄링이 알아서 흡수한다.
//
// 부수 효과: revIdx[off + lane] 을 32 lane이 동시에 읽어 연속 접근(coalesced)이 된다.
// 기존에는 인접 스레드가 서로 다른 노드의 명단을 읽어 완전히 흩어졌다.
// (clusterC/Sinv/Coef 는 i가 흩어져 있어 어느 쪽이든 gather다 — 여기는 안 바뀐다)
//
// 합칠 값은 4개 — 보정 합 3 + 받은 개수 1.
// ───────────────────────────────────────────────────────────────────────────
__global__ void xpbdVolumeGatherApplyWarpKernel(
	int N,
	const float3* pos_pred_in,
	float3* pos_pred_out,
	const float* invMass,
	const int* revOffset,
	const int* revCount,
	const int* revIdx,
	const float3* clusterC,
	const float* clusterSinv,
	const float* clusterCoef,
	const float* restLen,
	float invMassScale,
	float underRelax)
{
	const int warpsPerBlock = blockDim.x >> 5;
	const int lane = threadIdx.x & 31;
	const int idx = blockIdx.x * warpsPerBlock + (threadIdx.x >> 5); // 가우시안 = idx
	if (idx >= N) return;                                            // 워프 균일

	const float3 pi = pos_pred_in[idx];
	const float wi = invMass[idx] * invMassScale;
	if (wi <= 0.0f) { if (lane == 0) pos_pred_out[idx] = pi; return; } // 고정점 (워프 균일)

	const int off = revOffset[idx];
	const int cnt = revCount[idx];

	float3 sum = make_float3(0.0f, 0.0f, 0.0f);
	int received = 0;
	for (int k = lane; k < cnt; k += 32) {
		const int i = revIdx[off + k];
		if (i < 0 || i >= N) continue;
		const float coef = clusterCoef[i];
		if (coef == 0.0f) continue;
		const glm::mat3 Sinv = loadSym3(clusterSinv + i * 6);
		const float3 d = f3_sub(pi, clusterC[i]);
		const glm::vec3 g = Sinv * glm::vec3(d.x, d.y, d.z);
		const float3 dp = make_float3(g.x * wi * coef, g.y * wi * coef, g.z * wi * coef);
		if (!isfinite(dp.x) || !isfinite(dp.y) || !isfinite(dp.z)) continue;
		sum = f3_add(sum, dp);
		++received;
	}
	const float sx = warpAllSum(sum.x);
	const float sy = warpAllSum(sum.y);
	const float sz = warpAllSum(sum.z);
	const int rec = (int)(warpAllSum((float)received) + 0.5f);

	// 평균·클램프·쓰기는 lane 0만 (기존 B와 동일한 식)
	if (lane != 0) return;
	if (rec <= 0) { pos_pred_out[idx] = pi; return; }
	const float invRec = 1.0f / (float)rec;
	float3 avg = make_float3(sx * invRec, sy * invRec, sz * invRec);
	if (!isfinite(avg.x) || !isfinite(avg.y) || !isfinite(avg.z)) { pos_pred_out[idx] = pi; return; }

	const float rl = restLen[idx];
	if (rl > 1e-12f) {
		const float maxStep = 0.25f * rl;
		const float len = f3_len(avg);
		if (len > maxStep && len > 1e-20f) { atomicAdd(&g_clampCtr[1], 1u); avg = f3_mul(avg, maxStep / len); }
	}
	pos_pred_out[idx] = f3_add(pi, f3_mul(avg, underRelax));
}

// Squash: 상단 슬랩은 rest 위치에서 축 방향으로 curDisp만큼 이동한 위치에 pin,
// 하단 슬랩은 rest 위치에 pin. 둘 다 invMass=0으로 솔버가 못 건드리게 한다.
// 슬래브의 목표 위치. 위 슬래브만 하중축으로 curDisp 만큼 밀린다.
__device__ __forceinline__ float3 squashSlabTarget(
	bool isTop, int j,
	const float3* topRest, const float3* botRest,
	int axis, float curDisp)
{
	float3 r = isTop ? topRest[j] : botRest[j];
	if (isTop) {
		if (axis == 0) r.x -= curDisp;
		else if (axis == 1) r.y -= curDisp;
		else r.z -= curDisp;
	}
	return r;
}

// squash 하네스의 경계조건.
//   slip = 0 : no-slip 평판. 세 축을 모두 구속한다 (invMass = 0).
//   slip = 1 : 미끄럼 평판. 하중축만 구속하고 접선 두 축은 자유롭게 둔다.
//
// 미끄럼을 기본값으로 두는 이유: 그 경우 정확해가 균질 1축 변형이고 P1 사면체가 아핀
// 변위장을 정확히 표현하므로, FEM 참조해에 이산화 오차가 사실상 없다 (n=8 과 n=16 의
// 부피비가 소수 5자리까지 동일, 요소별 J 편차 ~1e-10, nu=0.499 에서 J=0.99978).
// no-slip 은 평판 모서리가 응력 집중이라 응답이 비균질하고(30%에서 J 0.80~1.03) 부피비가
// 메쉬에 따라 아직 움직인다. 두 솔버가 같은 조건이어야 비교가 성립하므로 뷰어도 같이 바꾼다.
__global__ void applySquashSlabKernel(
	int nTop, int nBot,
	const int* topIdx, const int* botIdx,
	const float3* topRest, const float3* botRest,
	float3* pos, float3* vel, float* invMass,
	int axis, float curDisp, int slip)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int nTotal = nTop + nBot;
	if (i >= nTotal) return;

	const bool isTop = (i < nTop);
	const int  j  = isTop ? i : (i - nTop);
	const int  id = isTop ? topIdx[i] : botIdx[j];
	const float3 r = squashSlabTarget(isTop, j, topRest, botRest, axis, curDisp);

	if (slip) {
		// invMass 는 건드리지 않는다. 매 프레임 1.0 으로 리셋되므로 제약이 슬래브를
		// 접선 방향으로 밀 수 있고, 그것이 곧 미끄럼이다.
		float3 p = pos[id];
		float3 v = vel[id];
		if (axis == 0) { p.x = r.x; v.x = 0.0f; }
		else if (axis == 1) { p.y = r.y; v.y = 0.0f; }
		else { p.z = r.z; v.z = 0.0f; }
		pos[id] = p;
		vel[id] = v;
	}
	else {
		pos[id] = r;
		vel[id] = make_float3(0.0f, 0.0f, 0.0f);
		invMass[id] = 0.0f;
	}
}

// 미끄럼 평판 전용. invMass 가 0 이 아니라 제약이 슬래브를 하중축으로도 밀어내므로,
// 규정 변위를 실제로 걸려면 매 반복 하중축만 되돌려야 한다. 접선 두 축은 건드리지 않는다.
__global__ void pinSquashSlabAxisKernel(
	int nTop, int nBot,
	const int* topIdx, const int* botIdx,
	const float3* topRest, const float3* botRest,
	float3* pos, int axis, float curDisp)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int nTotal = nTop + nBot;
	if (i >= nTotal) return;

	const bool isTop = (i < nTop);
	const int  j  = isTop ? i : (i - nTop);
	const int  id = isTop ? topIdx[i] : botIdx[j];
	const float3 r = squashSlabTarget(isTop, j, topRest, botRest, axis, curDisp);

	float3 p = pos[id];
	if (axis == 0) p.x = r.x;
	else if (axis == 1) p.y = r.y;
	else p.z = r.z;
	pos[id] = p;
}

// Press 모드 평판 접촉. groundProjectKernel 과 같은 규칙: 평판 밖으로 나간 중심만 하중축 좌표를 평판에 맞추고,
// 접선 두 축은 건드리지 않는다(마찰 없음 = 미끄럼 평판과 같은 경계조건). 잡은 점(w=0)은 제외.
__global__ void squashPressProjectKernel(int N, float3* pos, const float* invMass, int axis, float lo, float hi)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;
	if (invMass[idx] <= 0.0f) return;
	float3 p = pos[idx];
	const float v = (axis == 0) ? p.x : (axis == 1) ? p.y : p.z;
	if (!(v < lo) && !(v > hi)) return;   // 안쪽 (NaN 도 그대로 둔다)
	const float t = (v < lo) ? lo : hi;
	if (axis == 0) p.x = t;
	else if (axis == 1) p.y = t;
	else p.z = t;
	pos[idx] = p;
}

// 진단/시각화용: 각 볼륨 클러스터의 현재 J = sqrt(det Sigma_cur / det Sigma_rest)를 계산한다.
// 제약을 걸지 않고 '측정만' 한다. rest 상태에서 이 값이 1이어야 정상이다.
__global__ void computeVolumeJKernel(
	int N,
	const float3* pos_in,
	const int* offset,
	const int* nbrCount,
	const int* nbrIdx,
	const float* detRest,
	const int* matType,
	float* outJ)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	outJ[idx] = 1.0f;
	if (matType[idx] != 0) return;

	const int off = offset[idx];
	const int cnt = nbrCount[idx];

	float3 c = pos_in[idx];
	int used = 0;
	for (int k = 0; k < cnt; ++k) {
		const int j = nbrIdx[off + k];
		if (j < 0 || j >= N) continue;
		c = f3_add(c, pos_in[j]);
		++used;
	}
	if (used < 3) return;
	const float invN = 1.0f / (float)(used + 1);
	c = f3_mul(c, invN);

	glm::mat3 S(0.0f);
	{
		const float3 d = f3_sub(pos_in[idx], c);
		const glm::vec3 v(d.x, d.y, d.z);
		S += glm::outerProduct(v, v);
	}
	for (int k = 0; k < cnt; ++k) {
		const int j = nbrIdx[off + k];
		if (j < 0 || j >= N) continue;
		const float3 d = f3_sub(pos_in[j], c);
		const glm::vec3 v(d.x, d.y, d.z);
		S += glm::outerProduct(v, v);
	}
	S *= invN;

	float sCur, detHatCur;
	if (!sigmaScaleAndShapeDet(S, sCur, detHatCur)) return;
	const float detCur = sCur * sCur * sCur * detHatCur;
	const float dRest = detRest[idx];
	if (dRest < 1e-30f) return;
	const float J = sqrtf(detCur / dRest);
	if (isfinite(J) && J > 0.0f) outJ[idx] = J;
}

// [TN] 클러스터 J → per-Gaussian J 집계 (전치 CSR = '나를 포함하는 클러스터' 평균).
//  한 가우시안이 평균 h_j(≈22)개 클러스터에 겹쳐 속하므로 그 평균을 취한다.
//  (이 집계본의 공간 매끄러움은 [J-smooth] 진단으로 사전 검증됨: 상대거칠기 0.17~0.23)
__global__ void aggregateJPerGaussianKernel(
	int N, const float* clusterJ, const int* matType,
	const int* revOffset, const int* revCount, const int* revIdx, float* outJ)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	const int off = revOffset[i], cnt = revCount[i];
	float s = 0.0f; int used = 0;
	for (int e = 0; e < cnt; ++e) {
		const int L = revIdx[off + e];
		if (L < 0 || L >= N) continue;
		if (matType[L] != 0) continue;          // volume 클러스터만 J 가 유효
		const float j = clusterJ[L];
		if (!isfinite(j) || j <= 0.0f) continue;
		s += j; ++used;
	}
	outJ[i] = (used > 0) ? (s / (float)used) : 1.0f;   // h_j=0 (실측 34개) → J=1 폴백
}

__global__ void regionBalloonClearStatsKernel(float* stats)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < 9) stats[idx] = 0.0f;
}

__global__ void regionBalloonSumKernel(
	int count,
	const int* regionIdx,
	const float3* pos,
	float* stats)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= count) return;

	const int idx = regionIdx[tid];
	const float3 p = pos[idx];
	if (!isfinite(p.x) || !isfinite(p.y) || !isfinite(p.z)) return;

	atomicAdd(&stats[0], p.x);
	atomicAdd(&stats[1], p.y);
	atomicAdd(&stats[2], p.z);
}

__global__ void regionBalloonCovKernel(
	int count,
	const int* regionIdx,
	const float3* pos,
	float* stats)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= count) return;

	const float invCount = 1.0f / fmaxf((float)count, 1.0f);
	const float3 c = make_float3(stats[0] * invCount, stats[1] * invCount, stats[2] * invCount);

	const int idx = regionIdx[tid];
	const float3 d = f3_sub(pos[idx], c);
	if (!isfinite(d.x) || !isfinite(d.y) || !isfinite(d.z)) return;

	// Store covariance sums. The 1/count normalization is applied when reading.
	atomicAdd(&stats[3], d.x * d.x);
	atomicAdd(&stats[4], d.y * d.y);
	atomicAdd(&stats[5], d.z * d.z);
	atomicAdd(&stats[6], d.x * d.y);
	atomicAdd(&stats[7], d.x * d.z);
	atomicAdd(&stats[8], d.y * d.z);
}

__global__ void regionBalloonApplyKernel(
	int count,
	const int* regionIdx,
	const float3* pos_pred_in,
	float3* pos_pred_out,
	const float* invMass,
	const float* stats,
	float detRest,
	float restScale,
	float compliance,
	float strength,
	float maxStepRatio,
	float dt,
	float invMassScale,
	float underRelax)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= count) return;

	const int idx = regionIdx[tid];
	const float3 pi = pos_pred_in[idx];
	const float wi = invMass[idx] * invMassScale;
	if (wi <= 0.0f || detRest <= 1e-30f || restScale <= 1e-12f) {
		pos_pred_out[idx] = pi;
		return;
	}

	const float invCount = 1.0f / fmaxf((float)count, 1.0f);
	const float3 c = make_float3(stats[0] * invCount, stats[1] * invCount, stats[2] * invCount);

	glm::mat3 Scur(0.0f);
	Scur[0][0] = stats[3] * invCount;
	Scur[1][1] = stats[4] * invCount;
	Scur[2][2] = stats[5] * invCount;
	Scur[0][1] = Scur[1][0] = stats[6] * invCount;
	Scur[0][2] = Scur[2][0] = stats[7] * invCount;
	Scur[1][2] = Scur[2][1] = stats[8] * invCount;

	float sCur, detHatCur;
	if (!sigmaScaleAndShapeDet(Scur, sCur, detHatCur)) {
		pos_pred_out[idx] = pi;
		return;
	}
	const float detCur = sCur * sCur * sCur * detHatCur;
	const float J = sqrtf(detCur / detRest);
	if (!isfinite(J) || J <= 0.0f) {
		pos_pred_out[idx] = pi;
		return;
	}

	// C = J - 1. Compression (J < 1) should push particles outward.
	const float C = J - 1.0f;
	if (fabsf(C) < 1e-4f) {
		pos_pred_out[idx] = pi;
		return;
	}

	glm::mat3 Sinv;
	{
		glm::vec3 eig;
		glm::mat3 V;
		eigenDecomposition_glm(Scur, eig, V);
		const float lmax = fmaxf(eig.z, 1e-20f);
		const float lmin = 0.08f * lmax;
		const glm::vec3 e = glm::max(eig, glm::vec3(lmin));

		glm::mat3 D(0.0f);
		D[0][0] = e.x; D[1][1] = e.y; D[2][2] = e.z;
		const glm::mat3 Sclamped = V * D * glm::transpose(V);
		if (!inverse3x3_safe(Sclamped, Sinv, 1e-20f)) {
			pos_pred_out[idx] = pi;
			return;
		}
	}

	const float3 d = f3_sub(pi, c);
	const glm::vec3 n = Sinv * glm::vec3(d.x, d.y, d.z);
	float3 dir = make_float3(n.x, n.y, n.z);
	const float dirLen = f3_len(dir);
	if (dirLen <= 1e-12f || !isfinite(dirLen)) {
		pos_pred_out[idx] = pi;
		return;
	}
	dir = f3_mul(dir, 1.0f / dirLen);

	const float dt2 = fmaxf(dt * dt, 1e-8f);
	const float stiffness = 1.0f / (1.0f + compliance / dt2);
	float step = (-C) * strength * stiffness * restScale;
	const float maxStep = fmaxf(1e-6f, maxStepRatio * restScale);
	step = fminf(maxStep, fmaxf(-maxStep, step));

	const float3 dx = f3_mul(dir, step * wi * underRelax);
	pos_pred_out[idx] = f3_add(pi, dx);
}

static int computeTotalNeighbors(const FORWARD::ChainMail& cm)
{
	const int N = static_cast<int>(cm.numElements());
	int total = 0;
	for (int i = 0; i < N; ++i) {
		const auto& e = cm.getElement(i);
		const int end = e.offset + e.neighborCnt;
		if (end > total) total = end;
	}
	return total;
}

#include <thread>
#include <atomic>
#include <chrono>

// ===============================================================
// Step 3 — 물성 앵커: 클러스터별 compliance 채우기
//   α_vol(i) = c_vol / (λ_Lamé · V_rest(i))
// 큰 살덩이(V_rest 큼)는 강성 k = λ·V 가 커서 α가 작고(뻣뻣),
// 작은 덩이는 α가 커서(무름) — 제약 에너지 (1/2)·λ·V·(J−1)² 가
// 실제 그 부피의 재질을 누르는 에너지와 일치하도록 만드는 공식이다.
// ===============================================================
__global__ void fillPhysicalAlphaKernel(
	int N,
	const float* Vrest,
	const int* matType,
	float lambdaLame,
	float cvol,
	float* alpha_out)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	const float V = Vrest[idx];
	if (matType[idx] != 0 || V <= 1e-24f || lambdaLame <= 0.0f) {
		// 볼륨 물질이 아니면 커널 게이트가 어차피 거르지만, 안전하게 "사실상 무제약"을 넣어둔다.
		alpha_out[idx] = 1e6f;
		return;
	}
	alpha_out[idx] = cvol / (lambdaLame * V);
}

// 물성 파라미터가 바뀌거나 클러스터가 리빌드될 때 d_alpha_vol을 다시 채운다.
static void refreshPhysicalAlpha()
{
	const int N = cm_num_elements;
	if (N <= 0 || !d_alpha_vol || !d_V_rest || !d_matType) return;

	const float E = g_volMatE;
	const float nu = fminf(fmaxf(g_volMatNu, 0.0f), 0.49f); // ν→0.5에서 λ가 발산하므로 상한
	const float lambdaLame = E * nu / ((1.0f + nu) * (1.0f - 2.0f * nu));

	const int threads = 256;
	const int blocks = (N + threads - 1) / threads;
	fillPhysicalAlphaKernel << <blocks, threads >> > (
		N, d_V_rest, d_matType, lambdaLame, g_volCvol, d_alpha_vol);
	cudaDeviceSynchronize();

	if (g_volUsePhysicalAlpha) {
		// 채운 α의 스케일을 눈으로 확인한다. V_rest 4자릿수 스프레드 → α도 그만큼 벌어진다.
		std::vector<float> h_alpha(N);
		std::vector<int> h_mat(N);
		cudaMemcpy(h_alpha.data(), d_alpha_vol, sizeof(float) * N, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_mat.data(), d_matType, sizeof(int) * N, cudaMemcpyDeviceToHost);
		float aMin = FLT_MAX, aMax = 0.0f;
		double aSum = 0.0;
		int cnt = 0;
		for (int i = 0; i < N; ++i) {
			if (h_mat[i] != 0) continue;
			aMin = fminf(aMin, h_alpha[i]);
			aMax = fmaxf(aMax, h_alpha[i]);
			aSum += h_alpha[i];
			++cnt;
		}
		printf("[VolumeGaussian] physical alpha: E=%.3e nu=%.3f -> lambda_Lame=%.3e | c_vol=%.3f\n",
			E, nu, lambdaLame, g_volCvol);
		if (cnt > 0)
			printf("[VolumeGaussian] alpha_vol range: min=%.3e max=%.3e avg=%.3e (volume %d개)\n",
				aMin, aMax, aSum / cnt, cnt);
	}
}

// ===============================================================
// k-ring 볼륨 클러스터 빌드 (+ rest 사전계산 + rest-J 자기검증)
//
// 각 가우시안 i에 대해 그래프 BFS k-hop 멤버를 모아 볼륨 전용 CSR을 만든다.
// - k=1 이면 그래프 직접 이웃과 동일 → 기존 동작과 완전히 같다.
// - 멤버가 cap을 넘으면 BFS(깊이) 순서에서 균등 스트라이드로 서브샘플.
//   → 커널 비용/메모리 상한이 k와 무관하게 고정된다.
// 그래프 로드 시 1회 + UI에서 k 변경 시마다 호출된다.
// ===============================================================
// 부피 GS 용 클러스터 탐욕 컬러링. 클러스터 = {리더} ∪ members[리더].
// 두 클러스터가 가우시안 하나라도 공유하면 같은 색이 될 수 없다.
//   하한: 한 가우시안 j 를 포함하는 클러스터 h_j 개는 서로 전부 충돌 → 색 ≥ max h_j (공짜로 나온다)
//   비용: Σ_j h_j² (클러스터마다 멤버의 소속 클러스터를 전부 훑는다). 너무 크면 건너뛰고 하한만 보고한다.
static void buildVolumeGSColors(int N, const std::vector<std::vector<int>>& members,
	const std::vector<char>& isLeader, const std::vector<int>& revOff,
	const std::vector<int>& revCnt, const std::vector<int>& revIdx)
{
	if (d_volGSOrder) { cudaFree(d_volGSOrder); d_volGSOrder = nullptr; }
	g_volGSColorOffset.clear();
	g_volGSLowerBound = 0;
	g_volGSSkipped = false;
	if (N <= 0) return;

	const auto t0 = std::chrono::high_resolution_clock::now();
	double work = 0.0;
	for (int j = 0; j < N; ++j) {
		g_volGSLowerBound = std::max(g_volGSLowerBound, revCnt[j]);
		work += (double)revCnt[j] * revCnt[j];
	}
	std::vector<int> leaders;
	for (int i = 0; i < N; ++i) if (isLeader[i]) leaders.push_back(i);
	const int M = (int)leaders.size();
	if (M == 0) return;
	if (work > 4.0e9) {
		g_volGSSkipped = true;
		printf("[VolumeGS] coloring skipped: sum h_j^2 = %.3g too large | clusters %d | colors >= %d (max h_j)\n",
			work, M, g_volGSLowerBound);
		return;
	}

	std::vector<int> color(N, -1);
	std::vector<int> stamp;           // stamp[c] == 현재 클러스터 → 색 c 금지
	int numColors = 0;
	for (int l : leaders) {
		auto forbid = [&](int j) {
			for (int k = 0; k < revCnt[j]; ++k) {
				const int m = revIdx[revOff[j] + k];
				if (m >= 0 && m < N && color[m] >= 0) stamp[color[m]] = l;
			}
		};
		if ((int)stamp.size() < numColors + 1) stamp.resize(numColors + 1, -1);
		forbid(l);
		for (int j : members[l]) if (j >= 0 && j < N) forbid(j);
		int c = 0;
		while (c < numColors && stamp[c] == l) ++c;
		color[l] = c;
		numColors = std::max(numColors, c + 1);
	}

	g_volGSColorOffset.assign(numColors + 1, 0);
	for (int l : leaders) g_volGSColorOffset[color[l] + 1]++;
	for (int c = 0; c < numColors; ++c) g_volGSColorOffset[c + 1] += g_volGSColorOffset[c];
	std::vector<int> fill(g_volGSColorOffset.begin(), g_volGSColorOffset.end() - 1);
	std::vector<int> order(M);
	for (int l : leaders) order[fill[color[l]]++] = l;
	cudaMalloc(&d_volGSOrder, sizeof(int) * M);
	cudaMemcpy(d_volGSOrder, order.data(), sizeof(int) * M, cudaMemcpyHostToDevice);

	int minSize = M, maxSize = 0, under32 = 0;
	for (int c = 0; c < numColors; ++c) {
		const int s = g_volGSColorOffset[c + 1] - g_volGSColorOffset[c];
		minSize = std::min(minSize, s);
		maxSize = std::max(maxSize, s);
		if (s < 32) ++under32;
	}
	const double ms = std::chrono::duration<double, std::milli>(
		std::chrono::high_resolution_clock::now() - t0).count();
	printf("[VolumeGS] %d clusters -> %d colors (lower bound max h_j = %d) | clusters/color mean %.1f, min %d, max %d,"
		" colors with <32 clusters %d | build %.1f ms\n",
		M, numColors, g_volGSLowerBound, (double)M / numColors, minSize, maxSize, under32, ms);
}

static void rebuildVolumeClusters()
{
	const int N = cm_num_elements;
	if (N <= 0 || g_hGraphOffset.empty() || !d_pos_rest) return;

	const int k = g_volRingK;
	const int cap = g_volMaxMembers;
	const int leaderR = std::max(1, g_volLeaderMinHop);
	const auto t0 = std::chrono::high_resolution_clock::now();

	// rest 위치를 호스트로 (farthest-point 반원 선발 + 통계에 사용)
	std::vector<float3> h_rest(N);
	cudaMemcpy(h_rest.data(), d_pos_rest, sizeof(float3) * N, cudaMemcpyDeviceToHost);

	// isLeader는 아래에서 결정된다.
	//  - r=1: 모든 노드가 반장 (병렬 경로)
	//  - r≥2: 탐욕적 커버(greedy set cover) — 반장의 '실제 명단 멤버'로 covered를 표시하고,
	//         아직 어떤 명단에도 못 든 노드가 없어질 때까지 반장을 계속 뽑는다.
	//         → h_j=0(부피 보정 못 받는 노드)이 원리적으로 안 생긴다.
	std::vector<char> isLeader(N, 0);
	std::vector<std::vector<int>> members(N); // members[i] = 반장 i의 명단 (비반장은 빈 채로)

	// h_j (각 노드가 몇 개 명단에 들었나). 상한 U가 켜지면 멤버 선발이 이 값을 읽어
	// 이미 U겹 덮인 후보를 건너뛴다 → 순차 처리 필수(뒤 리더가 앞 리더의 결과를 봐야 함).
	std::vector<int> coverCnt(N, 0);
	const int coverMax = g_volCoverMax; // 0 = 상한 없음

	// 고차수 허브에서 BFS 폭주 방지.
	//
	// ★ 2048 → 8192 (dense benchmark, N=102,363)
	//   과거 주석은 "평균 차수 25 → 3홉이면 25³=15,600 > 2048 이라 잘린다"고 봤으나
	//   그것은 트리(지수 증가) 가정이었다. 얼굴은 2-manifold 표면이라 이웃끼리
	//   이웃을 공유하므로 k홉 공은 제곱으로 자란다 — 실측 회귀 결과 ∝ k^2.15.
	//     k=1:25.5  k=2:102  k=3:257  k=4:509  k=5:875  k=6:1341  (서브샘플 전 평균)
	//   2048에서의 절단율: k≤3 = 0.000%, k=4 = 0.028%, k=5 = 0.602%, k=6 = 9.492%
	//   → k≤4는 무해했으나 k=6이 10% 가까이 잘려 k 스윕 결과가 오염됐다.
	//   8192면 전 구간이 깨끗해진다. 비용은 그래프 로드 시 1회 BFS뿐(런타임 0).
	const int HARD_CAP = 8192;

	// ── [진단] HARD_CAP 절단 실태 ────────────────────────────────────────
	// 값을 바꿀 때마다 다시 재기 위해 계측을 남겨둔다. 추측하지 말고 센다.
	// 논문 Ablation의 "UI 슬라이더 k가 실제 반경과 일치하는가"에 대한 근거.
	// r=1 경로는 멀티스레드라 atomic이어야 한다.
	std::atomic<long long> diagCalls{ 0 };   // computeMembers 호출 수 (= 클러스터 수)
	std::atomic<long long> diagCapped{ 0 };  // HARD_CAP에 걸려 중도 절단된 수
	std::atomic<long long> diagOverCap{ 0 }; // cap(64) 초과로 서브샘플된 수
	std::atomic<long long> diagBallSum{ 0 }; // 서브샘플 '전' 공 크기 합
	std::atomic<long long> diagBallMax{ 0 }; // 서브샘플 '전' 공 크기 최대
	std::atomic<long long> diagHopSum{ 0 };  // 실제 도달 홉 합

	// 노드 i를 반장으로 삼아 k홉 BFS + cap 서브샘플로 명단을 만든다. 반환: 멤버 벡터.
	// stamp/frontier 버퍼는 호출자가 재사용하도록 넘긴다(할당 회피).
	auto computeMembers = [&](int i, std::vector<int>& stamp, std::vector<int>& frontier,
		std::vector<int>& nextFrontier, std::vector<int>& collected) -> std::vector<int> {
		collected.clear();
		bool hitCap = false; // [진단] HARD_CAP에 걸려 BFS가 홉 도중에 끊겼나
		int reachedHop = 0;  // [진단] 실제 도달 홉 (요청값 k와 일치하는지 확인용)
		// BFS는 홉 순서로 append하므로 각 홉의 후보가 collected 안에서 '연속 구간'을 이룬다.
		// 따라서 홉 번호를 원소마다 저장할 필요 없이 경계만 기록하면 된다(홉 계층 표집용).
		//   홉 d(1-based)의 구간 = [ hopEnd[d-2] , hopEnd[d-1] )   (d=1이면 [0, hopEnd[0]))
		int hopEnd[16] = { 0 };
		if (k <= 1) {
			const int off = g_hGraphOffset[i], cnt = g_hGraphCount[i];
			for (int e = 0; e < cnt; ++e) {
				const int j = g_hGraphIdx[off + e];
				if (j >= 0 && j < N && j != i) collected.push_back(j);
			}
			reachedHop = 1;
			hopEnd[0] = (int)collected.size();
		}
		else {
			frontier.clear();
			stamp[i] = i;
			frontier.push_back(i);
			for (int d = 0; d < k && !frontier.empty() && (int)collected.size() < HARD_CAP; ++d) {
				nextFrontier.clear();
				bool capped = false;
				for (int f : frontier) {
					const int off = g_hGraphOffset[f], cnt = g_hGraphCount[f];
					for (int e = 0; e < cnt; ++e) {
						const int j = g_hGraphIdx[off + e];
						if (j < 0 || j >= N || stamp[j] == i) continue;
						stamp[j] = i;
						collected.push_back(j);
						nextFrontier.push_back(j);
						if ((int)collected.size() >= HARD_CAP) { capped = true; break; }
					}
					if (capped) break;
				}
				frontier.swap(nextFrontier);
				reachedHop = d + 1;
				if (d < 16) hopEnd[d] = (int)collected.size(); // 이 홉까지의 누적 경계
				if (capped) { hitCap = true; break; }
			}
		}
		// ── h_j 상한 필터 ────────────────────────────────────────────────
		// 이미 U겹 덮인 후보는 명단에서 뺀다. 이게 상한을 만드는 유일한 지점이다.
		// 홉 계층 표집이 hopEnd 경계를 쓰므로 압축하면서 경계도 같이 다시 만든다.
		if (coverMax > 0 && !collected.empty()) {
			const int nH = (reachedHop < 16) ? reachedHop : 16;
			int newHopEnd[16] = { 0 };
			int w = 0, lo = 0;
			for (int dd = 0; dd < nH; ++dd) {
				const int hi = hopEnd[dd];
				for (int m = lo; m < hi; ++m)
					if (coverCnt[collected[m]] < coverMax) collected[w++] = collected[m];
				newHopEnd[dd] = w;
				lo = hi;
			}
			collected.resize(w);
			for (int dd = 0; dd < 16; ++dd) hopEnd[dd] = newHopEnd[dd];
		}

		// [진단] 서브샘플 '전'의 공 크기를 기록한다. cap(64)로 줄인 뒤에 재면
		// 전부 64로 보여서 절단 여부를 알 수 없다.
		{
			const long long ball = (long long)collected.size();
			diagCalls.fetch_add(1, std::memory_order_relaxed);
			diagBallSum.fetch_add(ball, std::memory_order_relaxed);
			diagHopSum.fetch_add(reachedHop, std::memory_order_relaxed);
			if (hitCap) diagCapped.fetch_add(1, std::memory_order_relaxed);
			if (ball > cap) diagOverCap.fetch_add(1, std::memory_order_relaxed);
			long long prev = diagBallMax.load(std::memory_order_relaxed);
			while (ball > prev && !diagBallMax.compare_exchange_weak(prev, ball, std::memory_order_relaxed)) {}
		}
		if ((int)collected.size() <= cap) return collected;
		// cap 초과 → 서브샘플
		if (g_volMemberSelect == 2) {
			// ── 홉 계층 farthest-point ──────────────────────────────────────
			// 전체 후보에서 그냥 FPS를 하면 '먼 것부터' 뽑히므로 바깥 테두리(=유사도가
			// 가장 먼 k홉 멤버)가 과대표집되고 1홉 핵심이 밀려난다. 유사도 그래프는
			// "가까울수록 같은 조직"이라고 말하는데 표본이 그 반대로 가는 셈이다.
			// 그래서 홉별로 예산을 먼저 배정하고 그 안에서만 FPS를 한다.
			//   1홉 : 전원          (가장 유사한 이웃은 통째로 보존)
			//   2..k: 남은 예산을 남은 홉 수로 균등 배분 (모자란 홉의 잔여는 뒤로 이월)
			// 예) cap=64, k=3, 후보 25/77/155 → 25 / 20 / 19
			//     cap=64, k=6, 후보 25/77/155/252/369/516 → 25 / 8 / 8 / 8 / 8 / 7
			// ※ md(이미 뽑힌 점까지의 최소거리)는 홉 경계를 넘어 전역으로 갱신한다.
			//   홉마다 독립으로 뽑으면 홉 사이에서 점이 겹쳐 공간 분산이 나빠진다.
			const int Mc = (int)collected.size();
			const float3 cpos = h_rest[i];
			auto dist2 = [](const float3& a, const float3& b) {
				const float dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
				return dx * dx + dy * dy + dz * dz;
			};
			std::vector<float> md(Mc);
			std::vector<char> taken(Mc, 0);
			for (int m = 0; m < Mc; ++m) md[m] = dist2(h_rest[collected[m]], cpos);

			const int nHop = (reachedHop < 16) ? reachedHop : 16;
			std::vector<int> pick; pick.reserve(cap);
			int remaining = cap;
			for (int d = 0; d < nHop && remaining > 0; ++d) {
				const int lo = (d == 0) ? 0 : hopEnd[d - 1];
				const int hi = hopEnd[d];
				if (hi <= lo) continue;
				const int hopsLeft = nHop - d;
				// d==0(1홉)은 전원. 그 뒤는 남은 예산 / 남은 홉 수 (올림).
				int quota = (d == 0) ? remaining : (remaining + hopsLeft - 1) / hopsLeft;
				if (quota > hi - lo) quota = hi - lo;
				for (int s = 0; s < quota; ++s) {
					int cur = -1; float best = -1.0f;
					for (int m = lo; m < hi; ++m) {
						if (taken[m]) continue;
						if (md[m] > best) { best = md[m]; cur = m; }
					}
					if (cur < 0) break;
					taken[cur] = 1;
					pick.push_back(collected[cur]);
					--remaining;
					const float3 cp = h_rest[collected[cur]];
					for (int m = 0; m < Mc; ++m) {
						if (taken[m]) continue;
						const float dd = dist2(h_rest[collected[m]], cp);
						if (dd < md[m]) md[m] = dd;
					}
				}
			}
			return pick;
		}
		else if (g_volMemberSelect == 1) {
			// farthest-point: 공간적으로 최대한 퍼진 cap개 (방향 편향 제거)
			const float3 cpos = h_rest[i];
			const int Mc = (int)collected.size();
			std::vector<float> md(Mc);
			std::vector<char> taken(Mc, 0);
			auto dist2 = [](const float3& a, const float3& b) {
				const float dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
				return dx * dx + dy * dy + dz * dz;
			};
			int cur = 0; float best = -1.0f;
			for (int m = 0; m < Mc; ++m) {
				const float dd = dist2(h_rest[collected[m]], cpos);
				md[m] = dd;
				if (dd > best) { best = dd; cur = m; }
			}
			std::vector<int> pick; pick.reserve(cap);
			for (int s = 0; s < cap; ++s) {
				taken[cur] = 1;
				pick.push_back(collected[cur]);
				const float3 cp = h_rest[collected[cur]];
				int nxt = -1; float nbest = -1.0f;
				for (int m = 0; m < Mc; ++m) {
					if (taken[m]) continue;
					const float dd = dist2(h_rest[collected[m]], cp);
					if (dd < md[m]) md[m] = dd;
					if (md[m] > nbest) { nbest = md[m]; nxt = m; }
				}
				if (nxt < 0) break;
				cur = nxt;
			}
			return pick;
		}
		else {
			std::vector<int> pick(cap);
			const double stride = (double)collected.size() / (double)cap;
			for (int t = 0; t < cap; ++t) pick[t] = collected[(size_t)(t * stride)];
			return pick;
		}
	};

	if (leaderR <= 1 && coverMax > 0) {
		// ── r=1 + h_j 상한: 순차 ──────────────────────────────────────────
		// 상한은 "이미 U겹 덮인 후보를 뺀다"로 만들어지는데, 그러려면 뒤 리더가
		// 앞 리더의 결과(coverCnt)를 봐야 한다. 병렬로 돌리면 서로의 갱신을 못 본다.
		// 빌드는 1회성이므로 순차 비용(코어 수만큼 느려짐)을 감수한다.
		std::fill(isLeader.begin(), isLeader.end(), (char)1);
		g_volLeaderCount = N;
		std::vector<int> stamp(N, -1), frontier, nextFrontier, collected;
		for (int i = 0; i < N; ++i) {
			members[i] = computeMembers(i, stamp, frontier, nextFrontier, collected);
			++coverCnt[i]; // 반장은 자기 반의 멤버이기도 하다
			for (int j : members[i]) if (j >= 0 && j < N) ++coverCnt[j];
		}
	}
	else if (leaderR <= 1) {
		// ── r=1: 모든 노드가 반장 (멀티스레드 병렬, 상한 없음) ──
		std::fill(isLeader.begin(), isLeader.end(), (char)1);
		g_volLeaderCount = N;
		const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
		std::atomic<int> nextChunk{ 0 };
		const int CHUNK = 512;
		auto worker = [&]() {
			std::vector<int> stamp(N, -1), frontier, nextFrontier, collected;
			for (;;) {
				const int start = nextChunk.fetch_add(CHUNK);
				if (start >= N) break;
				const int end = std::min(N, start + CHUNK);
				for (int i = start; i < end; ++i)
					members[i] = computeMembers(i, stamp, frontier, nextFrontier, collected);
			}
		};
		std::vector<std::thread> pool;
		for (unsigned t = 0; t < hw; ++t) pool.emplace_back(worker);
		for (auto& th : pool) th.join();
	}
	else {
		// ── r≥2: 탐욕적 커버(greedy set cover) — 순차, 최소 겹침 보장 ──
		// 반장을 뽑을 때마다 그 '실제 명단 멤버'의 커버 횟수를 +1 하고,
		// 커버 횟수가 target 미만인 노드가 없어질 때까지 반장을 계속 뽑는다.
		// ★ 종료 조건이 "최소 target겹"인 이유: 겹침이 1겹뿐인 부위는 급격 드래그로
		//   J가 튈 때 평균으로 눌러줄 이웃이 없어 발산한다(눈썹·머리카락 폭주 버그).
		//   target≥3이면 그 부위에도 방어용 이웃이 최소 3개 확보된다.
		const int targetCover = std::max(1, std::min(6, g_volLeaderMinHop));
		// coverCnt는 바깥에서 선언됐다(상한 필터가 computeMembers 안에서 읽으므로).
		std::vector<int> stamp(N, -1), frontier, nextFrontier, collected;
		int leaderCnt = 0;
		// 1차: leaderR 간격으로 성기게 반장 배치 (억제 stamp로 최소 간격 유지)
		std::vector<int> stampR(N, -1);
		for (int seed = 0; seed < N; ++seed) {
			if (coverCnt[seed] >= targetCover) continue;
			if (stampR[seed] == -2) continue;
			isLeader[seed] = 1;
			++leaderCnt;
			std::vector<int> mem = computeMembers(seed, stamp, frontier, nextFrontier, collected);
			++coverCnt[seed];
			for (int j : mem) if (j >= 0 && j < N) ++coverCnt[j];
			members[seed] = std::move(mem);
			if (leaderR >= 2) {
				frontier.clear(); stampR[seed] = -2; frontier.push_back(seed);
				for (int d = 0; d < leaderR - 1 && !frontier.empty(); ++d) {
					nextFrontier.clear();
					for (int f : frontier) {
						const int off = g_hGraphOffset[f], cnt = g_hGraphCount[f];
						for (int e = 0; e < cnt; ++e) {
							const int j = g_hGraphIdx[off + e];
							if (j < 0 || j >= N || stampR[j] == -2) continue;
							stampR[j] = -2;
							nextFrontier.push_back(j);
						}
					}
					frontier.swap(nextFrontier);
				}
			}
		}
		// 2차: 아직 target 겹에 못 미친 노드를 반장으로 승격 (억제 무시). 여러 번 반복해 보장.
		//   자기 자신을 반장 삼으면 자기 coverCnt는 무조건 +1 되므로 반드시 진행된다(무한루프 없음).
		bool progressed = true;
		while (progressed) {
			progressed = false;
			for (int i = 0; i < N; ++i) {
				if (coverCnt[i] >= targetCover) continue;
				if (isLeader[i]) continue; // 이미 반장인데도 부족하면 이웃 부실 → 더 뽑아도 소용없음(고립)
				isLeader[i] = 1;
				++leaderCnt;
				std::vector<int> mem = computeMembers(i, stamp, frontier, nextFrontier, collected);
				++coverCnt[i];
				for (int j : mem) if (j >= 0 && j < N) ++coverCnt[j];
				members[i] = std::move(mem);
				progressed = true;
			}
		}
		g_volLeaderCount = leaderCnt;
		// 최소/부족 커버 통계
		int under = 0, zero = 0;
		for (int i = 0; i < N; ++i) { if (coverCnt[i] == 0) ++zero; if (coverCnt[i] < targetCover) ++under; }
		printf("[VolumeGaussian] greedy cover: targetCover=%d | 미달 노드=%d (%.2f%%), h_j=0=%d (그래프 고립)\n",
			targetCover, under, N > 0 ? 100.0 * under / N : 0.0, zero);
	}

	printf("[VolumeGaussian] leader select: r=%d | leaders M=%d / N=%d (%.1f%%, 1/%.1f) | member-select=%s | 커버=명단멤버기준(h_j=0 제거)\n",
		leaderR, g_volLeaderCount, N, N > 0 ? 100.0 * g_volLeaderCount / N : 0.0,
		g_volLeaderCount > 0 ? (double)N / g_volLeaderCount : 1.0,
		(g_volMemberSelect == 2) ? "hop-stratified FPS" : (g_volMemberSelect == 1) ? "farthest-point" : "stride");

	// ── [진단] HARD_CAP 절단 실태 출력 ───────────────────────────────────
	// 절단 비율이 0에 가까우면 "UI 슬라이더 k = 실제 클러스터 반경"이 성립하고,
	// k 스윕 결과를 논문에 쓸 수 있다. 크면 홉 단위 공정 절단으로 고쳐야 한다.
	{
		const long long calls = diagCalls.load();
		if (calls > 0) {
			printf("[VolumeGaussian][진단] HARD_CAP 검사 | k=%d cap=%d HARD_CAP=%d | 클러스터 %lld개\n"
				"    · HARD_CAP 절단 : %lld개 (%.3f%%)   <- 0에 가까워야 k 슬라이더가 유효\n"
				"    · 도달 홉       : 평균 %.2f / 요청 %d\n"
				"    · 공 크기(서브샘플 전) : 평균 %.1f, 최대 %lld\n"
				"    · cap 초과 서브샘플     : %lld개 (%.1f%%)\n",
				k, cap, HARD_CAP, calls,
				diagCapped.load(), 100.0 * (double)diagCapped.load() / (double)calls,
				(double)diagHopSum.load() / (double)calls, k,
				(double)diagBallSum.load() / (double)calls, diagBallMax.load(),
				diagOverCap.load(), 100.0 * (double)diagOverCap.load() / (double)calls);
		}
	}

	// ── 2) CSR 평탄화 + 업로드 ──
	std::vector<int> h_off(N), h_cnt(N);
	long long total = 0;
	for (int i = 0; i < N; ++i) {
		h_off[i] = (int)total;
		h_cnt[i] = (int)members[i].size();
		total += h_cnt[i];
	}
	// ── [진단] h_j 상한이 명단을 얼마나 굶겼나 ────────────────────────────
	// 상한을 너무 조이면 클러스터가 cap을 못 채워 공분산 표본이 부족해진다
	// (표본 부족 → J 노이즈 증가). 이 숫자를 보고 U를 조절한다.
	if (coverMax > 0) {
		int leaders = 0, starved = 0, minCnt = INT_MAX;
		long long sumCnt = 0;
		for (int i = 0; i < N; ++i) {
			if (!isLeader[i]) continue;
			++leaders;
			const int c = h_cnt[i];
			sumCnt += c;
			if (c < minCnt) minCnt = c;
			if (c < cap) ++starved;
		}
		if (leaders > 0) {
			printf("[VolumeGaussian][진단] h_j 상한 U=%d | 명단 크기: 평균 %.1f / cap %d, 최소 %d"
				" | cap 미달 클러스터 %d개 (%.1f%%)\n",
				coverMax, (double)sumCnt / leaders, cap, (minCnt == INT_MAX ? 0 : minCnt),
				starved, 100.0 * starved / leaders);
		}
	}

	std::vector<int> h_flat((size_t)std::max<long long>(total, 1));
	for (int i = 0; i < N; ++i) {
		if (h_cnt[i] > 0)
			std::copy(members[i].begin(), members[i].end(), h_flat.begin() + h_off[i]);
	}

	if (d_volOffset) cudaFree(d_volOffset);
	if (d_volCount) cudaFree(d_volCount);
	if (d_volIdx) cudaFree(d_volIdx);
	cudaMalloc(&d_volOffset, sizeof(int) * N);
	cudaMalloc(&d_volCount, sizeof(int) * N);
	cudaMalloc(&d_volIdx, sizeof(int) * h_flat.size());
	cudaMemcpy(d_volOffset, h_off.data(), sizeof(int) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_volCount, h_cnt.data(), sizeof(int) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_volIdx, h_flat.data(), sizeof(int) * h_flat.size(), cudaMemcpyHostToDevice);

	// ── 2.5) 전치 CSR (gather 모드용): "노드 j를 멤버로 포함하는 클러스터 목록" ──
	// 정방향은 "클러스터 i의 멤버들", 전치는 그 반대 방향이다.
	// ★ 서브샘플 때문에 관계가 비대칭이라(i가 j를 뽑아도 j는 i를 안 뽑았을 수 있음)
	//   무향 그래프라도 전치를 명시적으로 만들어야 한다.
	// ★ 클러스터 i는 {i 자신} ∪ members(i) 이므로 self 엣지도 넣는다.
	{
		// self 엣지는 '반장'에게만 (non-leader는 클러스터가 없으므로).
		std::vector<int> revCnt(N, 0);
		for (int i = 0; i < N; ++i) if (isLeader[i]) ++revCnt[i]; // self
		for (int i = 0; i < N; ++i)
			for (int j : members[i])
				++revCnt[j];

		std::vector<int> revOff(N);
		long long revTotal = 0;
		for (int i = 0; i < N; ++i) {
			revOff[i] = (int)revTotal;
			revTotal += revCnt[i];
		}

		std::vector<int> revFlat((size_t)std::max<long long>(revTotal, 1));
		std::vector<int> cursor = revOff;
		for (int i = 0; i < N; ++i)
			if (isLeader[i]) revFlat[cursor[i]++] = i;  // self: 반장 i는 자기 클러스터의 멤버
		for (int i = 0; i < N; ++i)
			for (int j : members[i])
				revFlat[cursor[j]++] = i;      // j는 클러스터 i의 멤버

		if (d_volRevOffset) cudaFree(d_volRevOffset);
		if (d_volRevCount) cudaFree(d_volRevCount);
		if (d_volRevIdx) cudaFree(d_volRevIdx);
		cudaMalloc(&d_volRevOffset, sizeof(int) * N);
		cudaMalloc(&d_volRevCount, sizeof(int) * N);
		cudaMalloc(&d_volRevIdx, sizeof(int) * revFlat.size());
		cudaMemcpy(d_volRevOffset, revOff.data(), sizeof(int) * N, cudaMemcpyHostToDevice);
		cudaMemcpy(d_volRevCount, revCnt.data(), sizeof(int) * N, cudaMemcpyHostToDevice);
		cudaMemcpy(d_volRevIdx, revFlat.data(), sizeof(int) * revFlat.size(), cudaMemcpyHostToDevice);
		// [진단] per-Gaussian J 집계 측정을 위한 호스트 캐시 (리빌드 때만 갱신)
		g_hRevOffset = revOff;
		g_hRevCount = revCnt;
		g_hRevIdx = revFlat;
		buildVolumeGSColors(N, members, isLeader, revOff, revCnt, revFlat);
	}

	const float buildMs = std::chrono::duration<float, std::milli>(
		std::chrono::high_resolution_clock::now() - t0).count();
	// 워프 커널 자동 선택의 판단 근거 — '클러스터당' 평균 멤버 수(= n).
	// 노드당(total/N)이 아니라 리더당이어야 한다. 워프 효율은 lane이 몇 명을 맡느냐로 정해지므로.
	g_volAvgMembers = (g_volLeaderCount > 0) ? (float)((double)total / g_volLeaderCount) : 0.0f;

	printf("[VolumeGaussian] k-ring build: k=%d cap=%d | members total=%lld (avg %.1f/node, %.1f/cluster)"
		" | %.0f ms | %.1f MB | Kernel=%s\n",
		k, cap, total, N > 0 ? (double)total / N : 0.0, g_volAvgMembers,
		buildMs, total * 4.0 / 1048576.0,
		volUseWarpKernel() ? (volUseWarpOnePassKernel() ? "워프당 1-pass" : "워프당") : "스레드당");

	// ── 3) rest 공분산 사전계산 (새 클러스터 기준) ──
	// (아래 precompute가 matType을 채운 뒤에야 "유효 volume 클러스터" 기준 커버리지를 잴 수 있다.)
	const int threads = 256;
	const int blocks = (N + threads - 1) / threads;
	precomputeVolumeGaussianKernel << <blocks, threads >> > (
		N,
		d_pos_rest,
		d_volOffset,
		d_volCount,
		d_volIdx,
		d_offset, d_nbrCount, d_nbrIdx, // 원본 그래프 CSR (restLen 계산용, 모든 노드)
		g_gsScalesPtr,
		g_gsRotationsPtr,
		d_density,      // opacity (loadGraph에서 density = opacity로 채워짐)
		g_volMixtureRest ? 1 : 0,
		d_Sigma_rest,
		d_detSigmaRest,
		d_V_rest,
		d_volRestLen,
		d_matType,
		g_volAnisoThreshold);
	if (d_lambda_vol) cudaMemset(d_lambda_vol, 0, sizeof(float) * N);
	// Independent rest data for the experimental cluster hyperelastic solver.
	// It consumes the same immutable cluster membership and Sigma_rest, but owns
	// its inverse/rest-center cache and all runtime multipliers.
	if (d_gnhRestC && d_gnhRestSinv && d_gnhValid) {
		precomputeGaussianNHRestKernel << <blocks, threads >> > (
			N, d_pos_rest, d_volOffset, d_volCount, d_volIdx,
			d_Sigma_rest, d_matType, d_gnhRestC, d_gnhRestSinv, d_gnhValid);
	}
	if (d_lambda_gnhD) cudaMemset(d_lambda_gnhD, 0, sizeof(float) * N);
	if (d_lambda_gnhH) cudaMemset(d_lambda_gnhH, 0, sizeof(float) * N);
	if (d_gnhClusterCoefD) cudaMemset(d_gnhClusterCoefD, 0, sizeof(float) * N);
	if (d_gnhClusterCoefH) cudaMemset(d_gnhClusterCoefH, 0, sizeof(float) * N);
	cudaDeviceSynchronize();

	// ── 4) matType 통계 — 분류가 납득 가능한지 눈으로 확인한다 ──
	std::vector<int> h_matType(N);
	std::vector<float> h_Vrest(N);
	cudaMemcpy(h_matType.data(), d_matType, sizeof(int) * N, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_Vrest.data(), d_V_rest, sizeof(float) * N, cudaMemcpyDeviceToHost);
	g_hVRestCache = h_Vrest; // 부피 가중 J용 캐시. V_rest는 리빌드 때만 바뀐다(매 프레임 D2H 불필요)

	g_volMatCount[0] = g_volMatCount[1] = g_volMatCount[2] = 0;
	int cNeighborShort = 0, cNumericFail = 0; // [진단①] 3=이웃부족, 4=수치실패
	double vSum = 0.0;
	float vMin = FLT_MAX, vMax = 0.0f;
	int vCnt = 0;
	for (int i = 0; i < N; ++i) {
		const int t = h_matType[i];
		if (t >= 0 && t <= 2) ++g_volMatCount[t];
		else if (t == 3) ++cNeighborShort;
		else if (t == 4) ++cNumericFail;
		if (t == 0) {
			const float v = h_Vrest[i];
			vSum += v;
			vMin = fminf(vMin, v);
			vMax = fmaxf(vMax, v);
			++vCnt;
		}
	}
	const double invN100 = (N > 0) ? (100.0 / (double)N) : 0.0;
	printf("[VolumeGaussian] matType: volume=%d (%.1f%%), surface=%d (%.1f%%), fiber=%d (%.1f%%)\n",
		g_volMatCount[0], g_volMatCount[0] * invN100,
		g_volMatCount[1], g_volMatCount[1] * invN100,
		g_volMatCount[2], g_volMatCount[2] * invN100);
	// [진단①] 탈락 사유 분해 — "정밀도 실패가 실재하는가"의 직접 답.
	printf("[Diag/reason] 이웃부족(cnt<4)=%d (%.2f%%) | 수치실패(정밀도 detHat<=1e-6)=%d (%.3f%%) | 이방성-fiber=%d\n",
		cNeighborShort, cNeighborShort * invN100,
		cNumericFail, cNumericFail * invN100,
		g_volMatCount[2]);
	if (cNumericFail == 0)
		printf("[Diag/reason] → 정밀도 실패 0건. \"평평해서 부피가 안 잡힌다\"는 사실상 없음. surface/fiber는 전부 의도된 물질 분류.\n");
	if (vCnt > 0) {
		printf("[VolumeGaussian] V_rest (volume type): min=%.3e max=%.3e avg=%.3e (스프레드 %.0f배)\n",
			vMin, vMax, vSum / (double)vCnt, (vMin > 0.0f) ? (vMax / vMin) : 0.0);
	}
	else {
		printf("[VolumeGaussian] WARNING: volume 타입 클러스터가 하나도 없다. 부피 제약이 전혀 걸리지 않는다.\n");
	}

	// ── [진단③] 유효 volume 커버리지: 각 노드가 몇 개의 "유효 volume 클러스터"의 멤버인가 ──
	// h_j = |{ℓ : j ∈ C_ℓ, leader ℓ의 matType==0}|.  h_j=0 이면 그 점은 부피 보정을 전혀 못 받음.
	// 동시에 반장을 인덱스 기준 1/2·1/4·1/8로 가상 솎았을 때 커버리지가 어떻게 무너지는지 시뮬(보수적 상한).
	{
		auto coverageStats = [&](int stride) {
			std::vector<int> h(N, 0);
			for (int i = 0; i < N; ++i) {
				if (h_matType[i] != 0) continue;         // 유효 volume 반장만
				if (stride > 1 && (i % stride) != 0) continue; // 가상 솎아내기(인덱스 기준)
				++h[i];                                  // self
				for (int j : members[i]) if (j >= 0 && j < N) ++h[j];
			}
			// min / p5 / mean / zeroCount 계산
			std::vector<int> sorted = h;
			std::sort(sorted.begin(), sorted.end());
			long long sum = 0; int zero = 0;
			for (int v : h) { sum += v; if (v == 0) ++zero; }
			const int p5 = sorted[(size_t)(0.05 * (N - 1))];
			printf("[Diag/coverage] leader 1/%d: h_j  min=%d  p5=%d  mean=%.1f  max=%d  |  h_j=0 노드=%d (%.2f%%)\n",
				stride, sorted.front(), p5, N > 0 ? (double)sum / N : 0.0, sorted.back(),
				zero, zero * invN100);
		};
		printf("[Diag/coverage] (h_j = 각 노드가 속한 유효 volume 클러스터 수. h_j=0 이면 부피 보정 전혀 못 받음)\n");
		coverageStats(1);  // 현재 (전원 반장)
		coverageStats(2);  // 1/2 가상
		coverageStats(4);  // 1/4
		coverageStats(8);  // 1/8
	}

	// ── 5) rest-J 자기검증 ──
	// 지금 막 rest 위치로 detRest를 쟀으므로 rest에서 J는 정확히 1이어야 한다.
	// 1에서 벗어나면 rest 상태에서도 제약이 물체를 밀어 발산한다.
	if (d_volJScratch) {
		computeVolumeJKernel << <blocks, threads >> > (
			N, d_pos_rest, d_volOffset, d_volCount, d_volIdx, d_detSigmaRest, d_matType, d_volJScratch);
		cudaDeviceSynchronize();

		std::vector<float> h_J(N);
		cudaMemcpy(h_J.data(), d_volJScratch, sizeof(float) * N, cudaMemcpyDeviceToHost);

		double jSum = 0.0, jErrMax = 0.0;
		int jCnt = 0, jBad = 0;
		for (int i = 0; i < N; ++i) {
			if (h_matType[i] != 0) continue;
			const double e = fabs((double)h_J[i] - 1.0);
			jSum += h_J[i];
			jErrMax = fmax(jErrMax, e);
			if (e > 0.01) ++jBad;
			++jCnt;
		}
		if (jCnt > 0) {
			printf("[VolumeGaussian] rest-J self-check: avg=%.5f  max|J-1|=%.5f  (|J-1|>1%%: %d / %d = %.2f%%)\n",
				jSum / (double)jCnt, jErrMax, jBad, jCnt, 100.0 * jBad / (double)jCnt);
			if (jErrMax > 0.05)
				printf("[VolumeGaussian] WARNING: rest에서 J가 1과 크게 다르다. 부피 제약이 가만히 있는 물체를 밀어낸다.\n");
		}
	}

	// V_rest가 새로 계산되었으므로 물성 기반 α도 다시 채운다
	refreshPhysicalAlpha();

	g_volPrecomputed = true;
}

// 거리 GS 용 무향 간선 + 탐욕 간선 컬러링. 그래프 업로드 직후 1회.
// 같은 색 안에서 점을 공유하지 않는 것이 보장되어야 in-place 커널이 경쟁 없이 돈다(탐욕법 상한 2Δ-1색).
static void buildDistanceGSEdges(int N,
	const std::vector<int>& off, const std::vector<int>& cnt, const std::vector<int>& idx,
	const std::vector<float>& dist, const std::vector<float>& stiff)
{
	if (d_gsEdge) { cudaFree(d_gsEdge); d_gsEdge = nullptr; }
	if (d_gsRest) { cudaFree(d_gsRest); d_gsRest = nullptr; }
	if (d_gsStiff) { cudaFree(d_gsStiff); d_gsStiff = nullptr; }
	if (d_gsLambda) { cudaFree(d_gsLambda); d_gsLambda = nullptr; }
	g_gsColorOffset.clear();
	g_gsNumEdges = 0;
	g_gsOneWayEdges = 0;
	if (N <= 0 || idx.empty()) return;

	const auto t0 = std::chrono::high_resolution_clock::now();
	struct GSEdge { int a, b; float rest, stiff; int dup; };
	std::vector<GSEdge> dir;
	dir.reserve(idx.size());
	const int M = static_cast<int>(idx.size());
	for (int i = 0; i < N; ++i) {
		for (int k = 0; k < cnt[i]; ++k) {
			const int e = off[i] + k;
			if (e < 0 || e >= M) break;
			const int j = idx[e];
			if (j < 0 || j >= N || j == i) continue;
			dir.push_back({ std::min(i, j), std::max(i, j), dist[e], stiff[e], 1 });
		}
	}
	std::sort(dir.begin(), dir.end(), [](const GSEdge& x, const GSEdge& y) {
		return x.a != y.a ? x.a < y.a : x.b < y.b; });
	std::vector<GSEdge> und;
	und.reserve(dir.size());
	for (const GSEdge& g : dir) {
		if (!und.empty() && und.back().a == g.a && und.back().b == g.b) {
			und.back().rest += g.rest;
			und.back().stiff += g.stiff;
			und.back().dup++;
		}
		else {
			und.push_back(g);
		}
	}
	int oneWay = 0;
	for (GSEdge& g : und) {
		if (g.dup == 1) ++oneWay;
		g.rest /= float(g.dup);
		g.stiff /= float(g.dup);
	}
	const int E = static_cast<int>(und.size());
	if (E == 0) return;

	std::vector<int> deg(N, 0);
	for (const GSEdge& g : und) { deg[g.a]++; deg[g.b]++; }
	const int maxDeg = *std::max_element(deg.begin(), deg.end());
	const int words = (2 * maxDeg + 63) / 64;
	std::vector<unsigned long long> used(size_t(N) * words, 0ull);
	std::vector<int> color(E);
	int numColors = 0;
	for (int e = 0; e < E; ++e) {
		unsigned long long* ua = &used[size_t(und[e].a) * words];
		unsigned long long* ub = &used[size_t(und[e].b) * words];
		int c = -1;
		for (int w = 0; w < words && c < 0; ++w) {
			const unsigned long long freeBits = ~(ua[w] | ub[w]);
			if (!freeBits) continue;
			int bit = 0;
			while (!((freeBits >> bit) & 1ull)) ++bit;
			c = w * 64 + bit;
		}
		ua[c >> 6] |= 1ull << (c & 63);
		ub[c >> 6] |= 1ull << (c & 63);
		color[e] = c;
		numColors = std::max(numColors, c + 1);
	}

	// 색 순서로 계수 정렬 (색 안에서는 a 순서 유지 → Morton 순서라 접근이 모인다)
	g_gsColorOffset.assign(numColors + 1, 0);
	for (int e = 0; e < E; ++e) g_gsColorOffset[color[e] + 1]++;
	for (int c = 0; c < numColors; ++c) g_gsColorOffset[c + 1] += g_gsColorOffset[c];
	std::vector<int> fill(g_gsColorOffset.begin(), g_gsColorOffset.end() - 1);
	std::vector<int2> h_edge(E);
	std::vector<float> h_rest(E), h_st(E);
	for (int e = 0; e < E; ++e) {
		const int p = fill[color[e]]++;
		h_edge[p] = make_int2(und[e].a, und[e].b);
		h_rest[p] = und[e].rest;
		h_st[p] = und[e].stiff;
	}

	cudaMalloc(&d_gsEdge, sizeof(int2) * E);
	cudaMalloc(&d_gsRest, sizeof(float) * E);
	cudaMalloc(&d_gsStiff, sizeof(float) * E);
	cudaMalloc(&d_gsLambda, sizeof(float) * E);
	cudaMemcpy(d_gsEdge, h_edge.data(), sizeof(int2) * E, cudaMemcpyHostToDevice);
	cudaMemcpy(d_gsRest, h_rest.data(), sizeof(float) * E, cudaMemcpyHostToDevice);
	cudaMemcpy(d_gsStiff, h_st.data(), sizeof(float) * E, cudaMemcpyHostToDevice);
	cudaMemset(d_gsLambda, 0, sizeof(float) * E);
	g_gsNumEdges = E;
	g_gsOneWayEdges = oneWay;

	int minSize = E, maxSize = 0;
	for (int c = 0; c < numColors; ++c) {
		const int s = g_gsColorOffset[c + 1] - g_gsColorOffset[c];
		minSize = std::min(minSize, s);
		maxSize = std::max(maxSize, s);
	}
	const double ms = std::chrono::duration<double, std::milli>(
		std::chrono::high_resolution_clock::now() - t0).count();
	printf("[DistanceGS] %d directed -> %d undirected edges (one-way %d = %.1f%%), max degree %d, "
		"%d colors (edges/color %d..%d), build %.1f ms\n",
		int(dir.size()), E, oneWay, 100.0 * oneWay / E, maxDeg, numColors, minSize, maxSize, ms);
}

static void ensureChainmailGPU(const FORWARD::ChainMail& cm)
{
	const int N = static_cast<int>(cm.numElements());
	const int totalNeighbors = computeTotalNeighbors(cm);

	if (chainmailInit && N == cm_num_elements && totalNeighbors == cm_num_neighbors) {//초기화 1회만 하기
		// k-ring 변경 등으로 볼륨 클러스터만 다시 만들어야 하는 경우
		if (g_volTopoDirty) {
			g_volTopoDirty = false;
			rebuildVolumeClusters();
		}
		return;
	}

	if (d_pos_curr) cudaFree(d_pos_curr);
	if (d_pos_next) cudaFree(d_pos_next);
	if (d_pos_xpbd_tmp) cudaFree(d_pos_xpbd_tmp);
	if (d_pos_rest) cudaFree(d_pos_rest);
	if (d_vel) cudaFree(d_vel);
	if (d_density) cudaFree(d_density);
	if (d_invMass) cudaFree(d_invMass);
	if (d_invMassSaved) { cudaFree(d_invMassSaved); d_invMassSaved = nullptr; }   // 붙잡기 상태도 새 그래프와 함께 버린다
	if (d_objW) { cudaFree(d_objW); d_objW = nullptr; }
	g_attActive = false;
	g_attCount = 0;
	if (d_time_curr) cudaFree(d_time_curr);
	if (d_time_next) cudaFree(d_time_next);
	if (d_offset) cudaFree(d_offset);
	if (d_nbrCount) cudaFree(d_nbrCount);
	if (d_nbrIdx) cudaFree(d_nbrIdx);
	if (d_nbrDist) cudaFree(d_nbrDist);
	if (d_nbrStiff) cudaFree(d_nbrStiff);
	if (d_rest_cos) cudaFree(d_rest_cos);
	if (d_pair_next_idx) cudaFree(d_pair_next_idx);
	if (d_angle_dp_sum) cudaFree(d_angle_dp_sum);
	if (d_angle_dp_count) cudaFree(d_angle_dp_count);
	if (d_lambda_curr) cudaFree(d_lambda_curr);
	if (d_lambda_next) cudaFree(d_lambda_next);
	if (d_Sigma_rest) cudaFree(d_Sigma_rest);
	if (d_detSigmaRest) cudaFree(d_detSigmaRest);
	if (d_V_rest) cudaFree(d_V_rest);
	if (d_volRestLen) cudaFree(d_volRestLen);
	if (d_matType) cudaFree(d_matType);
	if (d_lambda_vol) cudaFree(d_lambda_vol);
	if (d_alpha_vol) cudaFree(d_alpha_vol);
	if (d_gnhRestC) { cudaFree(d_gnhRestC); d_gnhRestC = nullptr; }
	if (d_gnhRestSinv) { cudaFree(d_gnhRestSinv); d_gnhRestSinv = nullptr; }
	if (d_gnhValid) { cudaFree(d_gnhValid); d_gnhValid = nullptr; }
	if (d_lambda_gnhD) { cudaFree(d_lambda_gnhD); d_lambda_gnhD = nullptr; }
	if (d_lambda_gnhH) { cudaFree(d_lambda_gnhH); d_lambda_gnhH = nullptr; }
	if (d_gnhClusterF) { cudaFree(d_gnhClusterF); d_gnhClusterF = nullptr; }
	if (d_gnhClusterCoefD) { cudaFree(d_gnhClusterCoefD); d_gnhClusterCoefD = nullptr; }
	if (d_gnhClusterCoefH) { cudaFree(d_gnhClusterCoefH); d_gnhClusterCoefH = nullptr; }
	if (d_volJScratch) cudaFree(d_volJScratch);
	if (d_volJPerG) { cudaFree(d_volJPerG); d_volJPerG = nullptr; }
	// vol CSR은 rebuildVolumeClusters가 스스로 free/malloc 하므로 반드시 null로 되돌린다 (double-free 방지)
	if (d_volOffset) { cudaFree(d_volOffset); d_volOffset = nullptr; }
	if (d_volCount) { cudaFree(d_volCount); d_volCount = nullptr; }
	if (d_volIdx) { cudaFree(d_volIdx); d_volIdx = nullptr; }
	if (d_vol_dp_sum) { cudaFree(d_vol_dp_sum); d_vol_dp_sum = nullptr; }
	if (d_vol_dp_count) { cudaFree(d_vol_dp_count); d_vol_dp_count = nullptr; }
	if (d_volRevOffset) { cudaFree(d_volRevOffset); d_volRevOffset = nullptr; }
	if (d_volRevCount) { cudaFree(d_volRevCount); d_volRevCount = nullptr; }
	if (d_volRevIdx) { cudaFree(d_volRevIdx); d_volRevIdx = nullptr; }
	if (d_volClusterC) { cudaFree(d_volClusterC); d_volClusterC = nullptr; }
	if (d_volClusterSinv) { cudaFree(d_volClusterSinv); d_volClusterSinv = nullptr; }
	if (d_volClusterCoef) { cudaFree(d_volClusterCoef); d_volClusterCoef = nullptr; }
	if (d_regionBalloonIdx) cudaFree(d_regionBalloonIdx);
	if (d_regionBalloonStats) cudaFree(d_regionBalloonStats);
	if (d_active_map) cudaFree(d_active_map);
	if (d_next_map) cudaFree(d_next_map);
	if (d_active_count) cudaFree(d_active_count);
	if (d_best_from) cudaFree(d_best_from);

	cm_num_elements = N;
	cm_num_neighbors = totalNeighbors;

	cudaMalloc(&d_pos_curr, sizeof(float3) * N);
	cudaMalloc(&d_pos_next, sizeof(float3) * N);
	cudaMalloc(&d_pos_xpbd_tmp, sizeof(float3) * N);
	cudaMalloc(&d_pos_rest, sizeof(float3) * N);
	cudaMalloc(&d_vel, sizeof(float3) * N);
	cudaMalloc(&d_density, sizeof(float) * N);
	cudaMalloc(&d_invMass, sizeof(float) * N);
	cudaMalloc(&d_time_curr, sizeof(float) * N);
	cudaMalloc(&d_time_next, sizeof(float) * N);
	cudaMalloc(&d_offset, sizeof(int) * N);
	cudaMalloc(&d_nbrCount, sizeof(int) * N);
	cudaMalloc(&d_nbrIdx, sizeof(int) * totalNeighbors);
	cudaMalloc(&d_nbrDist, sizeof(float) * totalNeighbors);
	cudaMalloc(&d_nbrStiff, sizeof(float) * totalNeighbors);
	cudaMalloc(&d_rest_cos, sizeof(float) * totalNeighbors);
	cudaMalloc(&d_pair_next_idx, sizeof(int) * totalNeighbors);
	cudaMalloc(&d_angle_dp_sum, sizeof(float3) * N);
	cudaMalloc(&d_angle_dp_count, sizeof(int) * N);
	cudaMalloc(&d_lambda_curr, sizeof(float) * totalNeighbors);
	cudaMalloc(&d_lambda_next, sizeof(float) * totalNeighbors);
	cudaMalloc(&d_Sigma_rest, sizeof(float) * N * 6);
	cudaMalloc(&d_detSigmaRest, sizeof(float) * N);
	cudaMalloc(&d_V_rest, sizeof(float) * N);
	cudaMalloc(&d_volRestLen, sizeof(float) * N);
	cudaMalloc(&d_matType, sizeof(int) * N);
	cudaMalloc(&d_lambda_vol, sizeof(float) * N);
	cudaMalloc(&d_alpha_vol, sizeof(float) * N);
	cudaMalloc(&d_gnhRestC, sizeof(float3) * N);
	cudaMalloc(&d_gnhRestSinv, sizeof(float) * N * 6);
	cudaMalloc(&d_gnhValid, sizeof(unsigned char) * N);
	cudaMalloc(&d_lambda_gnhD, sizeof(float) * N);
	cudaMalloc(&d_lambda_gnhH, sizeof(float) * N);
	cudaMalloc(&d_gnhClusterF, sizeof(float) * N * 9);
	cudaMalloc(&d_gnhClusterCoefD, sizeof(float) * N);
	cudaMalloc(&d_gnhClusterCoefH, sizeof(float) * N);
	cudaMalloc(&d_volJScratch, sizeof(float) * N);
	// [TN] per-Gaussian J 버퍼 + 디바이스 포인터 심볼 (커널 시그니처 변경 없이 접근)
	cudaMalloc(&d_volJPerG, sizeof(float) * N);
	{
		const float one = 1.0f;
		// 초기값 1 (물리 전에는 변형 없음). memset은 0을 채우므로 커널 대신 host 루프 대신 fill.
		std::vector<float> ones(N, 1.0f);
		cudaMemcpy(d_volJPerG, ones.data(), sizeof(float) * N, cudaMemcpyHostToDevice);
		(void)one;
		const float* devPtr = d_volJPerG;
		cudaMemcpyToSymbol(g_dVolJPerG, &devPtr, sizeof(const float*));
	}
	cudaMalloc(&d_vol_dp_sum, sizeof(float3) * N);
	cudaMalloc(&d_vol_dp_count, sizeof(int) * N);
	// gather용 클러스터 파라미터 (크기 N 고정이라 여기서 잡는다. 전치 CSR은 rebuild가 관리)
	cudaMalloc(&d_volClusterC, sizeof(float3) * N);
	cudaMalloc(&d_volClusterSinv, sizeof(float) * N * 6);
	cudaMalloc(&d_volClusterCoef, sizeof(float) * N);
	cudaMemset(d_lambda_vol, 0, sizeof(float) * N);
	cudaMemset(d_lambda_gnhD, 0, sizeof(float) * N);
	cudaMemset(d_lambda_gnhH, 0, sizeof(float) * N);
	cudaMemset(d_gnhClusterCoefD, 0, sizeof(float) * N);
	cudaMemset(d_gnhClusterCoefH, 0, sizeof(float) * N);
	cudaMalloc(&d_regionBalloonIdx, sizeof(int) * N);
	cudaMalloc(&d_regionBalloonStats, sizeof(float) * 9);
	cudaMemset(d_regionBalloonIdx, 0, sizeof(int) * N);
	cudaMemset(d_regionBalloonStats, 0, sizeof(float) * 9);
	g_regionBalloonCount = 0;
	g_regionBalloonDetRest = 0.0f;
	g_regionBalloonRestScale = 0.0f;
	g_regionBalloonAnchor = -1;
	cudaMalloc(&d_active_map, sizeof(int) * N);
	cudaMalloc(&d_next_map, sizeof(int) * N);
	cudaMalloc(&d_active_count, sizeof(int));
	cudaMalloc(&d_best_from, sizeof(int) * N);
	cudaMemset(d_active_map, 0, sizeof(int) * N);
	cudaMemset(d_next_map, 0, sizeof(int) * N);
	cudaMemset(d_active_count, 0, sizeof(int));
	cudaMemset(d_best_from, 0xFF, sizeof(int) * N);
	cudaMemset(d_angle_dp_sum, 0, sizeof(float3) * N);
	cudaMemset(d_angle_dp_count, 0, sizeof(int) * N);
	cudaMemset(d_lambda_curr, 0, sizeof(float) * totalNeighbors);
	cudaMemset(d_lambda_next, 0, sizeof(float) * totalNeighbors);

	std::vector<float3> h_pos(N);
	std::vector<float3> h_vel(N);
	std::vector<float> h_density(N);
	std::vector<float> h_invMass(N);
	std::vector<float> h_time(N);
	std::vector<int> h_offset(N);
	std::vector<int> h_count(N);
	std::vector<int> h_idx(totalNeighbors);
	std::vector<float> h_dist(totalNeighbors);
	std::vector<float> h_stiff(totalNeighbors);
	// Angle-constraint precompute buffers (same indexing as h_idx / d_nbrIdx).
	// We keep legacy restDot (v1·v2) to match the original angle-constraint behavior.
	// Sentinel 2.0f is kept for invalid pairs and is ignored in kernel by validity checks.
	// Use cosine rest-angle in [-1, 1]. 2.0f is an invalid sentinel.
	std::vector<float> h_restCos(totalNeighbors, 2.0f);
	std::vector<int> h_pairNextIdx(totalNeighbors, -1);

	for (int i = 0; i < N; ++i) {
		const auto& e = cm.getElement(i);
		h_pos[i] = make_float3(e.pos.x, e.pos.y, e.pos.z);
		h_vel[i] = make_float3(e.vel.x, e.vel.y, e.vel.z);
		h_density[i] = e.density;
		h_invMass[i] = e.invMass;
		h_time[i] = e.time;
		h_offset[i] = e.offset;
		h_count[i] = e.neighborCnt;
	}
	for (int i = 0; i < totalNeighbors; ++i) {
		const auto& n = cm.getNeighbor(i);
		h_idx[i] = n.idx;
		h_dist[i] = n.dist;
		h_stiff[i] = n.st;
	}
	// Build "umbrella" pairing with angular ordering around each center.
	// This prevents random KNN order from creating crossed pairs and ghost torque.
	for (int i = 0; i < N; ++i) {
		const int off = h_offset[i];
		const int cnt = h_count[i];
		if (cnt < 2) continue;

		struct OrderedNbr {
			int edge;
			int nidx;
			float3 vec;
			float len;
			float angle;
		};

		const float3 p0 = h_pos[i];
		std::vector<OrderedNbr> samples;
		samples.reserve(cnt);

		for (int k = 0; k < cnt; ++k) {
			const int e = off + k;
			const int nidx = h_idx[e];
			if (nidx < 0 || nidx >= N) continue;

			const float3 pn = h_pos[nidx];
			const float3 v = make_float3(pn.x - p0.x, pn.y - p0.y, pn.z - p0.z);
			const float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
			if (len <= 1e-8f) continue;
			samples.push_back({ e, nidx, v, len, 0.0f });
		}
		if (samples.size() < 2) continue;

		// Local 2D basis (u, v) on tangent plane for angle sorting.
		size_t iU = 0;
		for (size_t s = 1; s < samples.size(); ++s) {
			if (samples[s].len > samples[iU].len) iU = s;
		}
		float3 u = make_float3(
			samples[iU].vec.x / samples[iU].len,
			samples[iU].vec.y / samples[iU].len,
			samples[iU].vec.z / samples[iU].len);

		float bestPerpL2 = -1.0f;
		float3 bestPerp = make_float3(0.0f, 0.0f, 0.0f);
		for (size_t s = 0; s < samples.size(); ++s) {
			const float du = samples[s].vec.x * u.x + samples[s].vec.y * u.y + samples[s].vec.z * u.z;
			const float3 perp = make_float3(
				samples[s].vec.x - du * u.x,
				samples[s].vec.y - du * u.y,
				samples[s].vec.z - du * u.z);
			const float l2 = perp.x * perp.x + perp.y * perp.y + perp.z * perp.z;
			if (l2 > bestPerpL2) {
				bestPerpL2 = l2;
				bestPerp = perp;
			}
		}

		float3 vAxis;
		if (bestPerpL2 > 1e-12f) {
			const float invL = 1.0f / sqrtf(bestPerpL2);
			vAxis = make_float3(bestPerp.x * invL, bestPerp.y * invL, bestPerp.z * invL);
		}
		else {
			// Fallback for near-collinear neighborhoods.
			const float3 a = (fabsf(u.x) < 0.9f) ? make_float3(1.0f, 0.0f, 0.0f) : make_float3(0.0f, 1.0f, 0.0f);
			const float3 cross = make_float3(
				u.y * a.z - u.z * a.y,
				u.z * a.x - u.x * a.z,
				u.x * a.y - u.y * a.x);
			const float cl2 = cross.x * cross.x + cross.y * cross.y + cross.z * cross.z;
			if (cl2 <= 1e-12f) continue;
			const float invL = 1.0f / sqrtf(cl2);
			vAxis = make_float3(cross.x * invL, cross.y * invL, cross.z * invL);
		}

		for (auto& s : samples) {
			const float x = s.vec.x * u.x + s.vec.y * u.y + s.vec.z * u.z;
			const float y = s.vec.x * vAxis.x + s.vec.y * vAxis.y + s.vec.z * vAxis.z;
			s.angle = atan2f(y, x);
		}
		std::sort(samples.begin(), samples.end(), [](const OrderedNbr& a, const OrderedNbr& b) {
			return a.angle < b.angle;
			});

		for (size_t s = 0; s < samples.size(); ++s) {
			const auto& cur = samples[s];
			const auto& nxt = samples[(s + 1) % samples.size()];
			const int e0 = cur.edge;

			// [추가된 안전장치] 두 이웃 사이의 각도 차이(Gap) 계산
			float angleDiff = nxt.angle - cur.angle;
			if (angleDiff < 0.0f) angleDiff += 2.0f * 3.14159265358979323846f; // 음수면 360도 더해줌 (마지막->처음 연결 시)

			// 각도 차이가 너무 크면(예: 120도 이상) 허공을 가로지르는 유령 간선이므로 짝짓기 포기!
			if (angleDiff > (2.0f * 3.14159265358979323846f / 3.0f)) { // 120도 기준 (필요시 조절)
				h_pairNextIdx[e0] = -1; // 연결 끊음
				h_restCos[e0] = 2.0f;   // GPU가 무시하도록 2.0f 세팅
				continue;
			}

			// Pair mapping for kernel: edge e0 uses (nbrIdx[e0], pairNextIdx[e0]).
			h_pairNextIdx[e0] = nxt.nidx;

			// Legacy metric: restDot = v1·v2 (length-coupled).
			// restCos = dot(normalize(v1), normalize(v2))
			const float3 n1 = make_float3(cur.vec.x / cur.len, cur.vec.y / cur.len, cur.vec.z / cur.len);
			const float3 n2 = make_float3(nxt.vec.x / nxt.len, nxt.vec.y / nxt.len, nxt.vec.z / nxt.len);
			float c = n1.x * n2.x + n1.y * n2.y + n1.z * n2.z;
			c = fminf(1.0f, fmaxf(-1.0f, c));
			h_restCos[e0] = c;
		}
	}

	cudaMemcpy(d_pos_curr, h_pos.data(), sizeof(float3) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pos_rest, h_pos.data(), sizeof(float3) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vel, h_vel.data(), sizeof(float3) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_density, h_density.data(), sizeof(float) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_invMass, h_invMass.data(), sizeof(float) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_time_curr, h_time.data(), sizeof(float) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_offset, h_offset.data(), sizeof(int) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_nbrCount, h_count.data(), sizeof(int) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_nbrIdx, h_idx.data(), sizeof(int) * totalNeighbors, cudaMemcpyHostToDevice);
	cudaMemcpy(d_nbrDist, h_dist.data(), sizeof(float) * totalNeighbors, cudaMemcpyHostToDevice);
	cudaMemcpy(d_nbrStiff, h_stiff.data(), sizeof(float) * totalNeighbors, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rest_cos, h_restCos.data(), sizeof(float) * totalNeighbors, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pair_next_idx, h_pairNextIdx.data(), sizeof(int) * totalNeighbors, cudaMemcpyHostToDevice);
	buildDistanceGSEdges(N, h_offset, h_count, h_idx, h_dist, h_stiff);

	// ── 부피 가우시안: k-ring 클러스터 빌드 + rest 사전계산 + 검증 ──
	// 그래프 호스트 사본을 보관해서, 이후 UI에서 k를 바꿀 때 그래프 재로드 없이 리빌드한다.
	g_hGraphOffset = h_offset;
	g_hGraphCount = h_count;
	g_hGraphIdx = h_idx;
	printf("[VolumeGaussian] precompute for %d gaussians (aniso threshold %.3f)\n", N, g_volAnisoThreshold);
	rebuildVolumeClusters();

	chainmailInit = true;
}

static void uploadChainmailPositions(const FORWARD::ChainMail& cm)
{
	const int N = static_cast<int>(cm.numElements());
	std::vector<float3> h_pos(N);
	std::vector<float> h_time(N);
	for (int i = 0; i < N; ++i) {
		const auto& e = cm.getElement(i);
		h_pos[i] = make_float3(e.pos.x, e.pos.y, e.pos.z);
		h_time[i] = e.time;
	}
	cudaMemcpy(d_pos_curr, h_pos.data(), sizeof(float3) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_time_curr, h_time.data(), sizeof(float) * N, cudaMemcpyHostToDevice);
}

static void uploadChainmailPositionsIndexed(const FORWARD::ChainMail& cm, const std::vector<int>& indices)
{
	const int N = static_cast<int>(cm.numElements());
	if (indices.empty() || N == 0) {
		return;
	}

	std::vector<int> uniq = indices;
	std::sort(uniq.begin(), uniq.end());
	uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());

	for (int idx : uniq) {
		if (idx < 0 || idx >= N) {
			continue;
		}
		const auto& e = cm.getElement(idx);
		const float3 p = make_float3(e.pos.x, e.pos.y, e.pos.z);
		cudaMemcpy(d_pos_curr + idx, &p, sizeof(float3), cudaMemcpyHostToDevice);
		cudaMemcpy(d_time_curr + idx, &e.time, sizeof(float), cudaMemcpyHostToDevice);
	}
}

static void downloadChainmailPositions(FORWARD::ChainMail& cm)
{
	const int N = static_cast<int>(cm.numElements());
	std::vector<float3> h_pos(N);
	std::vector<float> h_time(N);
	cudaMemcpy(h_pos.data(), d_pos_curr, sizeof(float3) * N, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_time.data(), d_time_curr, sizeof(float) * N, cudaMemcpyDeviceToHost);
	for (int i = 0; i < N; ++i) {
		auto& e = cm.getElement(i);
		e.pos = glm::vec3(h_pos[i].x, h_pos[i].y, h_pos[i].z);
		e.time = h_time[i];
	}
}

static double detSym3FromStats(const float* stats, int count)
{
	if (count <= 3) return 0.0;
	const double invCount = 1.0 / (double)count;
	const double xx = (double)stats[3] * invCount;
	const double yy = (double)stats[4] * invCount;
	const double zz = (double)stats[5] * invCount;
	const double xy = (double)stats[6] * invCount;
	const double xz = (double)stats[7] * invCount;
	const double yz = (double)stats[8] * invCount;
	return xx * (yy * zz - yz * yz)
		- xy * (xy * zz - yz * xz)
		+ xz * (xy * yz - yy * xz);
}

static float traceScaleFromStats(const float* stats, int count)
{
	if (count <= 3) return 0.0f;
	const float invCount = 1.0f / (float)count;
	const float trace = (stats[3] + stats[4] + stats[5]) * invCount;
	return sqrtf(fmaxf(trace / 3.0f, 1e-12f));
}

static void updateRegionBalloonFromSeeds(const FORWARD::ChainMail& cm, const std::vector<int>& seedIdx)
{
	if (!g_useRegionBalloon || seedIdx.empty() || !d_regionBalloonIdx || !d_regionBalloonStats || !d_pos_rest) {
		return;
	}

	const int N = static_cast<int>(cm.numElements());
	if (N <= 0) return;

	std::set<int> selectedNodes;
	std::queue<std::pair<int, int>> q;
	for (int idx : seedIdx) {
		if (idx < 0 || idx >= N) continue;
		if (selectedNodes.insert(idx).second) {
			q.push({ idx, 0 });
		}
	}

	while (!q.empty()) {
		const auto current = q.front();
		q.pop();
		const int curIdx = current.first;
		const int depth = current.second;
		if (depth >= g_regionBalloonHops) continue;

		const auto& elem = cm.getElement(curIdx);
		const int off = elem.offset;
		const int cnt = elem.neighborCnt;
		for (int k = 0; k < cnt; ++k) {
			const int nIdx = cm.getNeighbor(off + k).idx;
			if (nIdx < 0 || nIdx >= N) continue;
			if (selectedNodes.insert(nIdx).second) {
				q.push({ nIdx, depth + 1 });
			}
		}
	}

	std::vector<int> h_idx;
	h_idx.reserve(selectedNodes.size());
	for (int idx : selectedNodes) {
		h_idx.push_back(idx);
	}

	const int count = static_cast<int>(h_idx.size());
	if (count < 4) {
		g_regionBalloonCount = 0;
		g_regionBalloonDetRest = 0.0f;
		g_regionBalloonRestScale = 0.0f;
		g_regionBalloonAnchor = -1;
		return;
	}

	cudaMemcpy(d_regionBalloonIdx, h_idx.data(), sizeof(int) * count, cudaMemcpyHostToDevice);

	const int threads = 256;
	const int blocks = (count + threads - 1) / threads;
	regionBalloonClearStatsKernel << <1, 32 >> > (d_regionBalloonStats);
	regionBalloonSumKernel << <blocks, threads >> > (count, d_regionBalloonIdx, d_pos_rest, d_regionBalloonStats);
	regionBalloonCovKernel << <blocks, threads >> > (count, d_regionBalloonIdx, d_pos_rest, d_regionBalloonStats);

	float h_stats[9] = {};
	cudaMemcpy(h_stats, d_regionBalloonStats, sizeof(float) * 9, cudaMemcpyDeviceToHost);

	const double detRest = detSym3FromStats(h_stats, count);
	const float restScale = traceScaleFromStats(h_stats, count);
	if (!(detRest > 1e-30) || !(restScale > 1e-12f)) {
		g_regionBalloonCount = 0;
		g_regionBalloonDetRest = 0.0f;
		g_regionBalloonRestScale = 0.0f;
		g_regionBalloonAnchor = -1;
		return;
	}

	g_regionBalloonCount = count;
	g_regionBalloonDetRest = (float)detRest;
	g_regionBalloonRestScale = restScale;
	g_regionBalloonAnchor = seedIdx.front();
}

//static void applySeedCommandsGPU(FORWARD::ChainMail& cm)
//{
//	const int N = static_cast<int>(cm.numElements());
//	if (N <= 0) return;
//
//	if (g_useActiveMap && g_activeMapReset) {
//		cudaMemset(d_active_map, 0, sizeof(int) * N);
//		cudaMemset(d_next_map, 0, sizeof(int) * N);
//		g_activeMapReset = false;
//	}
//
//	std::vector<int> cmdIdx;
//	std::vector<glm::vec3> cmdDelta;
//	drainGpuCommands(cmdIdx, cmdDelta);
//	if (cmdIdx.empty()) return;
//
//	struct Cmd { int idx; glm::vec3 delta; };
//	std::vector<Cmd> cmds;
//	cmds.reserve(cmdIdx.size());
//	for (size_t i = 0; i < cmdIdx.size(); ++i) {
//		const int idx = cmdIdx[i];
//		if (idx < 0 || idx >= N) continue;
//		cmds.push_back({ idx, cmdDelta[i] });
//	}
//	if (cmds.empty()) return;
//
//	std::sort(cmds.begin(), cmds.end(), [](const Cmd& a, const Cmd& b) {
//		return a.idx < b.idx;
//	});
//
//	std::vector<int> h_idx;
//	std::vector<float3> h_delta;
//	h_idx.reserve(cmds.size());
//	h_delta.reserve(cmds.size());
//
//	Cmd current = cmds[0];
//	for (size_t i = 1; i < cmds.size(); ++i) {
//		if (cmds[i].idx == current.idx) {
//			current.delta += cmds[i].delta;
//		}
//		else {
//			h_idx.push_back(current.idx);
//			h_delta.push_back(make_float3(current.delta.x, current.delta.y, current.delta.z));
//			current = cmds[i];
//		}
//	}
//	h_idx.push_back(current.idx);
//	h_delta.push_back(make_float3(current.delta.x, current.delta.y, current.delta.z));
//
//	if (g_useActiveMap) {
//		const int threads = 256;
//		const int blocksN = (N + threads - 1) / threads;
//		resetTimeKernel << <blocksN, threads >> > (N, d_time_curr, 1e9f);
//		resetTimeKernel << <blocksN, threads >> > (N, d_time_next, 1e9f);
//		cudaMemset(d_active_map, 0, sizeof(int) * N);
//		cudaMemset(d_next_map, 0, sizeof(int) * N);
//	}
//
//	int* d_idx = nullptr;
//	float3* d_delta = nullptr;
//	const int count = static_cast<int>(h_idx.size());
//	cudaMalloc(&d_idx, sizeof(int) * count);
//	cudaMalloc(&d_delta, sizeof(float3) * count);
//	cudaMemcpy(d_idx, h_idx.data(), sizeof(int) * count, cudaMemcpyHostToDevice);
//	cudaMemcpy(d_delta, h_delta.data(), sizeof(float3) * count, cudaMemcpyHostToDevice);
//
//	const int threads = 256;
//	const int blocks = (count + threads - 1) / threads;
//	applySeedCommandsKernel << <blocks, threads >> > (
//		count,
//		d_idx,
//		d_delta,
//		d_pos_curr,
//		d_time_curr,
//		g_useActiveMap ? d_active_map : nullptr);
//	cudaDeviceSynchronize();
//
//	cudaFree(d_idx);
//	cudaFree(d_delta);
//}
static int g_currentAnchorIdx = -1;
// 매 프레임 할당을 피하기 위한 정적 버퍼
static int* d_idx_static = nullptr;
static float3* d_delta_static = nullptr;
static const int MAX_SEEDS = 10000; 

static void applySeedCommandsGPU(FORWARD::ChainMail& cm)
{
    const int N = static_cast<int>(cm.numElements());
    if (N <= 0) return;

    // 1. 명령어 가져오기
    std::vector<int> cmdIdx;
    std::vector<glm::vec3> cmdDelta;
    drainGpuCommands(cmdIdx, cmdDelta);

    // 명령어가 없으면 앵커 해제 후 종료
    if (cmdIdx.empty()) {
        g_currentAnchorIdx = -1;
        return;
    }

    // 2. BFS 이웃 확장 (XPBD 방식)
    
    std::set<int> selectedNodes;
    std::queue<std::pair<int, int>> q;
    glm::vec3 totalDelta(0.0f);

    for (size_t i = 0; i < cmdIdx.size(); ++i) {
        int idx = cmdIdx[i];
        if (idx >= 0 && idx < N) {
            if (selectedNodes.insert(idx).second) {
                q.push({ idx, 0 });
                totalDelta += cmdDelta[i];
            }
        }
    }

    if (selectedNodes.empty()) return;
    
    // 평균 이동량 및 대표 앵커 설정
    glm::vec3 avgDelta = totalDelta / (float)cmdIdx.size();
    g_currentAnchorIdx = cmdIdx[0]; 

    while (!q.empty()) {
        auto current = q.front(); q.pop();
        int curIdx = current.first;
        int depth = current.second;

        if (depth >= bfsHops) continue;

        const auto& elem = cm.getElement(curIdx);
        for (int k = 0; k < elem.neighborCnt; ++k) {
            int nIdx = cm.getNeighbor(elem.offset + k).idx;
            if (nIdx >= 0 && nIdx < N && selectedNodes.insert(nIdx).second) {
                q.push({ nIdx, depth + 1 });
            }
        }
    }

    // 3. [수정됨] BFS로 모은 모든 이웃을 h_idx에 담기
    std::vector<int> h_idx;
    std::vector<float3> h_delta;
    h_idx.reserve(selectedNodes.size());
    h_delta.reserve(selectedNodes.size());

    float3 f3Delta = make_float3(avgDelta.x, avgDelta.y, avgDelta.z);
    for (int nodeIdx : selectedNodes) {
        h_idx.push_back(nodeIdx);
        h_delta.push_back(f3Delta); // 모두 같은 평균 이동량 적용
    }

    // 4. GPU 전송 및 커널 실행
    const int count = static_cast<int>(h_idx.size());
    
    // 정적 버퍼 초기화 (1회)
    if (!d_idx_static) {
        cudaMalloc(&d_idx_static, sizeof(int) * MAX_SEEDS);
        cudaMalloc(&d_delta_static, sizeof(float3) * MAX_SEEDS);
    }

    int safeCount = std::min(count, MAX_SEEDS);
    cudaMemcpy(d_idx_static, h_idx.data(), sizeof(int) * safeCount, cudaMemcpyHostToDevice);
    cudaMemcpy(d_delta_static, h_delta.data(), sizeof(float3) * safeCount, cudaMemcpyHostToDevice);

    if (g_useActiveMap) {
        const int threads = 256;
        const int blocksN = (N + threads - 1) / threads;
        resetTimeKernel << <blocksN, threads >> > (N, d_time_curr, 1e9f);
        cudaMemset(d_active_map, 0, sizeof(int) * N);
    }

    const int threads = 256;
    const int blocks = (safeCount + threads - 1) / threads;
    applySeedCommandsKernel << <blocks, threads >> > (
        safeCount,
        d_idx_static,
        d_delta_static,
        d_pos_curr,
        d_time_curr,
        g_useActiveMap ? d_active_map : nullptr);
    
    cudaDeviceSynchronize();
}
static void applySeedCommands(FORWARD::ChainMail& cm)
{
	const int N = static_cast<int>(cm.numElements());
	if (N <= 0) return;

	const int threads = 256;
	const int blocksN = (N + threads - 1) / threads;
	xpbdResetInvMassKernel << <blocksN, threads >> > (N, d_invMass, 1.0f);

	std::vector<int> cmdIdx;
	std::vector<glm::vec3> cmdDelta;
	drainGpuCommands(cmdIdx, cmdDelta);
	if (cmdIdx.empty()) {
		return;
	}

	std::vector<int> seedIdx;
	seedIdx.reserve(cmdIdx.size());
	glm::vec3 avgDelta(0.0f);
	int validCmds = 0;
	for (size_t i = 0; i < cmdIdx.size(); ++i) {
		const int idx = cmdIdx[i];
		if (idx < 0 || idx >= N) {
			continue;
		}
		seedIdx.push_back(idx);
		avgDelta += cmdDelta[i];
		++validCmds;
	}
	if (validCmds == 0) {
		return;
	}
	avgDelta *= (1.0f / float(validCmds));

	// Region Balloon은 피킹 seed 주변의 더 넓은 BFS 덩어리를 하나의 부피 타원체로 본다.
	// 기존 soft-selection pin은 그대로 유지하고, 부피 보존 region만 별도 hop으로 갱신한다.
	updateRegionBalloonFromSeeds(cm, seedIdx);
	
	std::set<int> selectedNodes;
	std::queue<std::pair<int, int>> q;
	for (int idx : seedIdx) {
		if (selectedNodes.insert(idx).second) {
			q.push({ idx, 0 });
		}
	}

	while (!q.empty()) {
		const auto current = q.front();
		q.pop();
		const int curIdx = current.first;
		const int depth = current.second;
		if (depth >= bfsHops) {
			continue;
		}

		const auto& elem = cm.getElement(curIdx);
		const int off = elem.offset;
		const int cnt = elem.neighborCnt;
		for (int k = 0; k < cnt; ++k) {
			const int nIdx = cm.getNeighbor(off + k).idx;
			if (nIdx < 0 || nIdx >= N) {
				continue;
			}
			if (selectedNodes.insert(nIdx).second) {
				q.push({ nIdx, depth + 1 });
			}
		}
	}

	std::vector<int> h_idx;
	std::vector<float3> h_delta;
	h_idx.reserve(selectedNodes.size());
	h_delta.reserve(selectedNodes.size());

	const float3 avgDelta3 = make_float3(avgDelta.x, avgDelta.y, avgDelta.z);
	for (int idx : selectedNodes) {
		h_idx.push_back(idx);
		h_delta.push_back(avgDelta3);
	}

	int* d_idx = nullptr;
	float3* d_delta = nullptr;
	const int count = static_cast<int>(h_idx.size());
	if (count <= 0) {
		return;
	}
	cudaMalloc(&d_idx, sizeof(int) * count);
	cudaMalloc(&d_delta, sizeof(float3) * count);
	cudaMemcpy(d_idx, h_idx.data(), sizeof(int) * count, cudaMemcpyHostToDevice);
	cudaMemcpy(d_delta, h_delta.data(), sizeof(float3) * count, cudaMemcpyHostToDevice);

	const int blocks = (count + threads - 1) / threads;
	applySeedCommandsKernel << <blocks, threads >> > (
		count,
		d_idx,
		d_delta,
		d_pos_curr,
		d_vel,
		d_invMass,
		d_time_curr);
	cudaDeviceSynchronize();

	cudaFree(d_idx);
	cudaFree(d_delta);
}

// ── Object mode 호스트 단계 ─────────────────────────────────────────────────
// 왜 필요한가: 로컬 제약(1-ring 거리·형상)만으로는 중력 하중이 물체 전체로 전달되지 않는다.
// Jacobi는 반복당 ~1홉만 전파하고, 표면 위주 점 분포는 모든 엣지 길이를 지킨 채로도 천처럼 접힌다.
// CPU 복제 실험 (scratchpad collapse_experiment.py, 셸 N=1500, 바닥 낙하 후 높이/원래높이):
//   거리만 0.03 · 형상 blend 1.0 0.37 · 중력 ×¼ 0.50 · 컴플라이언스 0 0.55 · 10 substep 0.55
//   · underRelax 1.0 0.21 (효과 없음) · 전역 shape matching 0.9~1.0 (넘어짐 제외)
// 반발: 입자 단위는 e와 무관하게 튕겼다(e=0.3/0.8 rise 0.53/0.49 — 투영 부산물).
//   물체 단위 충격량은 구 셸·감쇠 0에서 e=0 → 0.00, 0.3 → 0.08, 0.8 → 0.62 (이론 e² 0/0.09/0.64).
// 비용: 프레임당 N×float3 다운로드 1~3회 + O(N) double 루프 + 가벼운 커널 2개. 100프레임마다 실측 출력.
// ── 물체 = 그래프 연결 성분 ─────────────────────────────────────────────────
// 한 장면에 여러 물체가 있으면(pillow: 바구니·접시·쿠션들) 물체마다 형상 유지·반발을 따로 해야 한다.
// 예전에는 N개 전체를 한 강체로 맞춰서 쿠션들이 아래 물체와 한 몸처럼 붙어 움직였다.
// 그래프가 물체 사이를 끊어 두었다는 전제에서 연결 성분 = 물체. g_objMinComponent 미만 성분(floater)은
// 물체로 보지 않는다(id −1: 형상 유지·물체 반발 없음, 바닥 접촉·자기충돌은 입자 단위로 그대로).
static int objFindRoot(std::vector<int>& parent, int x)
{
	while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
	return x;
}

static bool objectEnsureComponents(int N)
{
	if (g_objCompN == N && g_objCompSrc == d_nbrIdx && g_objCompMinUsed == g_objMinComponent && d_objId)
		return g_objNumComponents > 0;
	if (!d_offset || !d_nbrCount || !d_nbrIdx || cm_num_neighbors <= 0 || N <= 0) return false;
	std::vector<int> off(N), cnt(N), nbr(cm_num_neighbors);
	cudaMemcpy(off.data(), d_offset, sizeof(int) * N, cudaMemcpyDeviceToHost);
	cudaMemcpy(cnt.data(), d_nbrCount, sizeof(int) * N, cudaMemcpyDeviceToHost);
	cudaMemcpy(nbr.data(), d_nbrIdx, sizeof(int) * cm_num_neighbors, cudaMemcpyDeviceToHost);

	std::vector<int> parent(N);
	for (int i = 0; i < N; ++i) parent[i] = i;
	for (int i = 0; i < N; ++i) {
		for (int k = 0; k < cnt[i]; ++k) {
			const int e = off[i] + k;
			if (e < 0 || e >= cm_num_neighbors) continue;
			const int j = nbr[e];
			if (j < 0 || j >= N) continue;
			const int ri = objFindRoot(parent, i), rj = objFindRoot(parent, j);
			if (ri != rj) parent[ri] = rj;
		}
	}
	std::vector<int> compSize(N, 0);
	for (int i = 0; i < N; ++i) ++compSize[objFindRoot(parent, i)];

	// 루트 → 물체 id (−1 미정, −2 작은 성분). 입자 순서대로 매겨 실행마다 같은 번호가 나온다.
	std::vector<int> rootId(N, -1);
	int K = 0, nSmall = 0, pSmall = 0;
	g_objIdHost.assign(N, -1);
	for (int i = 0; i < N; ++i) {
		const int r = objFindRoot(parent, i);
		if (rootId[r] == -1) {
			if (compSize[r] >= g_objMinComponent) rootId[r] = K++;
			else { rootId[r] = -2; ++nSmall; pSmall += compSize[r]; }
		}
		if (rootId[r] >= 0) g_objIdHost[i] = rootId[r];
	}
	g_objCount.assign(K, 0);
	for (int i = 0; i < N; ++i) if (g_objIdHost[i] >= 0) ++g_objCount[g_objIdHost[i]];
	std::vector<int> sorted(g_objCount);
	std::sort(sorted.begin(), sorted.end(), [](int a, int b) { return a > b; });
	for (int t = 0; t < 3; ++t) g_objLargest[t] = (t < K) ? sorted[t] : 0;

	if (d_objId && g_objCompN != N) { cudaFree(d_objId); d_objId = nullptr; }
	if (!d_objId) cudaMalloc(&d_objId, sizeof(int) * N);
	cudaMemcpy(d_objId, g_objIdHost.data(), sizeof(int) * N, cudaMemcpyHostToDevice);
	if (K > g_objParamCap) {
		if (d_objParams) cudaFree(d_objParams);
		if (d_objImpulse) cudaFree(d_objImpulse);
		cudaMalloc(&d_objParams, sizeof(float) * 16 * K);
		cudaMalloc(&d_objImpulse, sizeof(float) * 10 * K);
		g_objParamCap = K;
	}
	g_objPrevVW.assign(6 * (size_t)K, 0.0);
	g_objPrevOk.assign(K, 0);
	g_objNumComponents = K;
	g_objNumSmall = nSmall;
	g_objSmallParticles = pSmall;
	g_objCompN = N;
	g_objCompSrc = d_nbrIdx;
	g_objCompMinUsed = g_objMinComponent;
	g_objRestN = 0;   // 물체별 rest 무게중심을 다시 잰다
	printf("[Object] bodies = graph components >= %d pts: %d (largest %d / %d / %d) | small components ignored: %d (%d pts)\n",
		g_objMinComponent, K, g_objLargest[0], g_objLargest[1], g_objLargest[2], nSmall, pSmall);
	return K > 0;
}

static void objectEnsureRestCache(int N)
{
	if (g_objRestN == N && g_objRestSrc == d_pos_rest && (int)g_objRestRel.size() == 3 * N) return;
	const int K = g_objNumComponents;
	std::vector<float3> rest(N);
	cudaMemcpy(rest.data(), d_pos_rest, sizeof(float3) * N, cudaMemcpyDeviceToHost);
	std::vector<double> sum(3 * (size_t)K, 0.0);
	for (int i = 0; i < N; ++i) {
		const int id = g_objIdHost[i];
		if (id < 0) continue;
		sum[3 * id] += rest[i].x; sum[3 * id + 1] += rest[i].y; sum[3 * id + 2] += rest[i].z;
	}
	g_objC0.assign(3 * (size_t)K, 0.0f);
	for (int k = 0; k < K; ++k)
		for (int a = 0; a < 3; ++a)
			g_objC0[3 * k + a] = (float)(sum[3 * k + a] / std::max(g_objCount[k], 1));
	g_objRestRel.assign(3 * (size_t)N, 0.0);
	for (int i = 0; i < N; ++i) {
		const int id = g_objIdHost[i];
		if (id < 0) continue;
		// 커널이 float로 계산하는 (x_rest − c0)와 같은 값을 쓴다
		g_objRestRel[3 * i + 0] = (double)(rest[i].x - g_objC0[3 * id]);
		g_objRestRel[3 * i + 1] = (double)(rest[i].y - g_objC0[3 * id + 1]);
		g_objRestRel[3 * i + 2] = (double)(rest[i].z - g_objC0[3 * id + 2]);
	}
	g_objRestN = N;
	g_objRestSrc = d_pos_rest;
	g_objPrevValid = false;

	// GPU 경로용 사본: 호스트 경로와 같은 q·c0·n. Σq 는 float c0 때문에 정확히 0 이 아니므로 그대로 넘긴다.
	if (N > g_objGpuRestCap) {
		if (d_objRestRel) cudaFree(d_objRestRel);
		cudaMalloc(&d_objRestRel, sizeof(double) * 3 * (size_t)N);
		g_objGpuRestCap = N;
	}
	cudaMemcpy(d_objRestRel, g_objRestRel.data(), sizeof(double) * 3 * (size_t)N, cudaMemcpyHostToDevice);
	if (K > g_objGpuCap) {
		if (d_objSums) cudaFree(d_objSums);
		if (d_objShift) cudaFree(d_objShift);
		if (d_objSumQ) cudaFree(d_objSumQ);
		if (d_objC0Dev) cudaFree(d_objC0Dev);
		if (d_objCountDev) cudaFree(d_objCountDev);
		if (d_objDisp) cudaFree(d_objDisp);
		cudaMalloc(&d_objSums, sizeof(double) * 16 * K);   // 붙잡기 중에는 물체당 16 개
		cudaMalloc(&d_objShift, sizeof(double) * 3 * K);
		cudaMalloc(&d_objSumQ, sizeof(double) * 3 * K);
		cudaMalloc(&d_objC0Dev, sizeof(float) * 3 * K);
		cudaMalloc(&d_objCountDev, sizeof(int) * K);
		cudaMalloc(&d_objDisp, sizeof(float) * K);
		g_objGpuCap = K;
	}
	std::vector<double> sumQ(3 * (size_t)K, 0.0), shift0(3 * (size_t)K, 0.0);
	for (int i = 0; i < N; ++i) {
		const int id = g_objIdHost[i];
		if (id < 0) continue;
		for (int a = 0; a < 3; ++a) sumQ[3 * id + a] += g_objRestRel[3 * (size_t)i + a];
	}
	for (int t = 0; t < 3 * K; ++t) shift0[t] = (double)g_objC0[t];
	cudaMemcpy(d_objSumQ, sumQ.data(), sizeof(double) * 3 * K, cudaMemcpyHostToDevice);
	cudaMemcpy(d_objShift, shift0.data(), sizeof(double) * 3 * K, cudaMemcpyHostToDevice);
	cudaMemcpy(d_objC0Dev, g_objC0.data(), sizeof(float) * 3 * K, cudaMemcpyHostToDevice);
	cudaMemcpy(d_objCountDev, g_objCount.data(), sizeof(int) * K, cudaMemcpyHostToDevice);
	g_objShiftValid = false;
}

static bool objectInvert3(const double m[9], double inv[9])
{
	const double a = m[0], b = m[1], c = m[2], d = m[3], e = m[4], f = m[5], g = m[6], h = m[7], i = m[8];
	const double C00 = e * i - f * h, C01 = -(d * i - f * g), C02 = d * h - e * g;
	const double det = a * C00 + b * C01 + c * C02;
	const double scale = fabs(a) + fabs(e) + fabs(i);
	if (!(fabs(det) > 1e-12 * scale * scale * scale)) return false;
	const double s = 1.0 / det;
	inv[0] = C00 * s; inv[1] = -(b * i - c * h) * s; inv[2] = (b * f - c * e) * s;
	inv[3] = C01 * s; inv[4] = (a * i - c * g) * s;  inv[5] = -(a * f - c * d) * s;
	inv[6] = C02 * s; inv[7] = -(a * h - b * g) * s; inv[8] = (a * e - b * d) * s;
	return true;
}

// 물체 단위 반발: 접촉 입자들의 중심을 접촉점으로 보고 강체 충돌 충격량을 물체 전체에 준다.
//   u_pre = n·(v_prev + ω_prev × r) − g·dt   (충돌 전 접촉점 법선속도, 직전 프레임 강체 속도로 추정)
//   j = (−e·u_pre − u_now) / (1/M + (r×n)ᵀ I⁻¹ (r×n)),   Δv = j n / M,  Δω = I⁻¹ (r × j n)
// 정지 접촉(|u_pre| ≤ 2g·dt)은 건너뛴다. 입자 질량은 모두 1. 물체(연결 성분)마다 따로 계산한다.
static void objectBodyContact(int N, int blocks, int threads, float dt)
{
	const int K = g_objNumComponents;
	g_objHostPos.resize(N);
	g_objHostVel.resize(N);
	cudaMemcpy(g_objHostPos.data(), d_pos_curr, sizeof(float3) * N, cudaMemcpyDeviceToHost);
	cudaMemcpy(g_objHostVel.data(), d_vel, sizeof(float3) * N, cudaMemcpyDeviceToHost);
	if (!g_objPrevValid) std::fill(g_objPrevOk.begin(), g_objPrevOk.end(), (char)0);

	const double n[3] = { g_groundN[0], g_groundN[1], g_groundN[2] };
	const double planeD = (double)g_groundHeight + (double)g_groundRadius;
	std::vector<double> c(3 * (size_t)K, 0.0), vc(3 * (size_t)K, 0.0), L(3 * (size_t)K, 0.0);
	std::vector<double> RR(6 * (size_t)K, 0.0), pc(3 * (size_t)K, 0.0);   // RR: Σ r rᵀ (xx yy zz xy xz yz)
	std::vector<int> nc(K, 0);
	for (int i = 0; i < N; ++i) {
		const int id = g_objIdHost[i];
		if (id < 0) continue;
		const float3& x = g_objHostPos[i];
		const float3& v = g_objHostVel[i];
		c[3 * id] += x.x; c[3 * id + 1] += x.y; c[3 * id + 2] += x.z;
		vc[3 * id] += v.x; vc[3 * id + 1] += v.y; vc[3 * id + 2] += v.z;
	}
	for (int k = 0; k < K; ++k) {
		const double inv = 1.0 / std::max(g_objCount[k], 1);
		for (int a = 0; a < 3; ++a) { c[3 * k + a] *= inv; vc[3 * k + a] *= inv; }
	}
	for (int i = 0; i < N; ++i) {
		const int id = g_objIdHost[i];
		if (id < 0) continue;
		const float3& x = g_objHostPos[i];
		const float3& v = g_objHostVel[i];
		const double rx = x.x - c[3 * id], ry = x.y - c[3 * id + 1], rz = x.z - c[3 * id + 2];
		L[3 * id] += ry * v.z - rz * v.y;
		L[3 * id + 1] += rz * v.x - rx * v.z;
		L[3 * id + 2] += rx * v.y - ry * v.x;
		double* rr = &RR[6 * id];
		rr[0] += rx * rx; rr[1] += ry * ry; rr[2] += rz * rz;
		rr[3] += rx * ry; rr[4] += rx * rz; rr[5] += ry * rz;
		if (n[0] * x.x + n[1] * x.y + n[2] * x.z - planeD <= (double)g_groundSlop) {
			pc[3 * id] += x.x; pc[3 * id + 1] += x.y; pc[3 * id + 2] += x.z;
			++nc[id];
		}
	}
	const auto mul3 = [](const double M[9], const double a[3], double out[3]) {
		out[0] = M[0] * a[0] + M[1] * a[1] + M[2] * a[2];
		out[1] = M[3] * a[0] + M[4] * a[1] + M[5] * a[2];
		out[2] = M[6] * a[0] + M[7] * a[1] + M[8] * a[2];
	};
	const auto cross3 = [](const double a[3], const double b[3], double out[3]) {
		out[0] = a[1] * b[2] - a[2] * b[1];
		out[1] = a[2] * b[0] - a[0] * b[2];
		out[2] = a[0] * b[1] - a[1] * b[0];
	};
	const double g = g_groundGravity;
	const double e = g_groundRestitution;
	std::vector<float> imp(10 * (size_t)K, 0.0f);
	bool anyFired = false;
	for (int k = 0; k < K; ++k) {
		if (g_objCount[k] < 4) { g_objPrevOk[k] = 0; continue; }
		const double M = (double)g_objCount[k];
		const double* rr = &RR[6 * k];
		const double tr = rr[0] + rr[1] + rr[2];
		const double I[9] = {
			tr - rr[0], -rr[3], -rr[4],
			-rr[3], tr - rr[1], -rr[5],
			-rr[4], -rr[5], tr - rr[2] };
		double Iinv[9] = { 0.0 };
		const bool hasI = objectInvert3(I, Iinv);
		double w[3] = { 0.0, 0.0, 0.0 };
		if (hasI) mul3(Iinv, &L[3 * k], w);
		double v3[3] = { vc[3 * k], vc[3 * k + 1], vc[3 * k + 2] };

		if (nc[k] > 0 && g_objPrevOk[k]) {
			const double rc[3] = { pc[3 * k] / nc[k] - c[3 * k], pc[3 * k + 1] / nc[k] - c[3 * k + 1], pc[3 * k + 2] / nc[k] - c[3 * k + 2] };
			const double* pv = &g_objPrevVW[6 * k];
			const double* pw = &g_objPrevVW[6 * k + 3];
			double t[3];
			cross3(pw, rc, t);
			const double uPre = n[0] * (pv[0] + t[0]) + n[1] * (pv[1] + t[1]) + n[2] * (pv[2] + t[2]) - g * dt;
			cross3(w, rc, t);
			const double uNow = n[0] * (v3[0] + t[0]) + n[1] * (v3[1] + t[1]) + n[2] * (v3[2] + t[2]);
			if (uPre < -2.0 * g * dt && uNow < -e * uPre) {
				double rxn[3], Irxn[3] = { 0.0, 0.0, 0.0 };
				cross3(rc, n, rxn);
				if (hasI) mul3(Iinv, rxn, Irxn);
				const double Keff = 1.0 / M + rxn[0] * Irxn[0] + rxn[1] * Irxn[1] + rxn[2] * Irxn[2];
				const double j = (-e * uPre - uNow) / Keff;
				const double dv[3] = { n[0] * j / M, n[1] * j / M, n[2] * j / M };
				const double dw[3] = { Irxn[0] * j, Irxn[1] * j, Irxn[2] * j };   // I⁻¹ (r × j n) = j · I⁻¹ (r × n)
				float* J = &imp[10 * k];
				for (int a = 0; a < 3; ++a) {
					J[a] = (float)c[3 * k + a];
					J[3 + a] = (float)dv[a];
					J[6 + a] = (float)dw[a];
					v3[a] += dv[a];
					w[a] += dw[a];
				}
				J[9] = 1.0f;
				anyFired = true;
				++g_objImpulseCount;
			}
		}
		for (int a = 0; a < 3; ++a) { g_objPrevVW[6 * k + a] = v3[a]; g_objPrevVW[6 * k + 3 + a] = w[a]; }
		g_objPrevOk[k] = 1;
	}
	g_objPrevValid = true;
	if (anyFired) {
		cudaMemcpy(d_objImpulse, imp.data(), sizeof(float) * 10 * K, cudaMemcpyHostToDevice);
		objectAddRigidVelocityKernel << <blocks, threads >> > (N, d_pos_curr, d_vel, d_invMass, d_objId, d_objImpulse);
	}
}

// 전역 shape matching: predict 직후 위치로 A = Σ (p − c)(x_rest − c0)ᵀ 를 double 누적 → SVD 극분해.
// 1차 실험에서 반복마다 적용(s=0.1)과 프레임당 1회(s=0.5~1.0)가 같은 결과라 1회만 한다.
static void objectShapeMatch(int N, int blocks, int threads)
{
	const int K = g_objNumComponents;
	g_objHostPos.resize(N);
	cudaMemcpy(g_objHostPos.data(), d_pos_next, sizeof(float3) * N, cudaMemcpyDeviceToHost);
	std::vector<double> c(3 * (size_t)K, 0.0), A(9 * (size_t)K, 0.0);
	for (int i = 0; i < N; ++i) {
		const int id = g_objIdHost[i];
		if (id < 0) continue;
		const float3& p = g_objHostPos[i];
		c[3 * id] += p.x; c[3 * id + 1] += p.y; c[3 * id + 2] += p.z;
	}
	for (int k = 0; k < K; ++k) {
		const double inv = 1.0 / std::max(g_objCount[k], 1);
		for (int a = 0; a < 3; ++a) c[3 * k + a] *= inv;
	}
	// 자기충돌 활성 제한용: 물체별 경계상자(predict 위치)와 프레임 간 무게중심 이동량.
	{
		std::vector<float> boxes(6 * (size_t)K);
		for (int k = 0; k < K; ++k) {
			boxes[6 * k] = boxes[6 * k + 1] = boxes[6 * k + 2] = FLT_MAX;
			boxes[6 * k + 3] = boxes[6 * k + 4] = boxes[6 * k + 5] = -FLT_MAX;
		}
		for (int i = 0; i < N; ++i) {
			const int id = g_objIdHost[i];
			if (id < 0) continue;
			const float3& p = g_objHostPos[i];
			float* b = &boxes[6 * id];
			b[0] = fminf(b[0], p.x); b[1] = fminf(b[1], p.y); b[2] = fminf(b[2], p.z);
			b[3] = fmaxf(b[3], p.x); b[4] = fmaxf(b[4], p.y); b[5] = fmaxf(b[5], p.z);
		}
		const bool prevOk = (g_objPrevCom.size() == 3 * (size_t)K);
		double maxDisp = 0.0;
		if (prevOk) {
			for (int k = 0; k < K; ++k) {
				const double dx = c[3 * k] - g_objPrevCom[3 * k], dy = c[3 * k + 1] - g_objPrevCom[3 * k + 1], dz = c[3 * k + 2] - g_objPrevCom[3 * k + 2];
				maxDisp = std::max(maxDisp, sqrt(dx * dx + dy * dy + dz * dz));
			}
		}
		g_objPrevCom.assign(c.begin(), c.end());
		if (K > g_objBoxCap) {
			if (d_objBoxes) cudaFree(d_objBoxes);
			cudaMalloc(&d_objBoxes, sizeof(float) * 6 * K);
			g_objBoxCap = K;
		}
		cudaMemcpy(d_objBoxes, boxes.data(), sizeof(float) * 6 * K, cudaMemcpyHostToDevice);
		g_objMaxBodyDisp = (float)maxDisp;
		g_objBoxesValid = prevOk && std::isfinite(maxDisp);   // 첫 프레임은 이동량을 모르므로 제한하지 않는다
	}
	for (int i = 0; i < N; ++i) {
		const int id = g_objIdHost[i];
		if (id < 0) continue;
		const float3& p = g_objHostPos[i];
		const double dx = p.x - c[3 * id], dy = p.y - c[3 * id + 1], dz = p.z - c[3 * id + 2];
		const double* q = &g_objRestRel[3 * (size_t)i];
		double* a = &A[9 * id];
		a[0] += dx * q[0]; a[1] += dx * q[1]; a[2] += dx * q[2];
		a[3] += dy * q[0]; a[4] += dy * q[1]; a[5] += dy * q[2];
		a[6] += dz * q[0]; a[7] += dz * q[1]; a[8] += dz * q[2];
	}
	const auto det3 = [](const Eigen::Matrix3d& M) {
		return M(0, 0) * (M(1, 1) * M(2, 2) - M(1, 2) * M(2, 1))
			- M(0, 1) * (M(1, 0) * M(2, 2) - M(1, 2) * M(2, 0))
			+ M(0, 2) * (M(1, 0) * M(2, 1) - M(1, 1) * M(2, 0));
	};
	std::vector<float> params(16 * (size_t)K, 0.0f);
	for (int k = 0; k < K; ++k) {
		if (g_objCount[k] < 4) continue;
		const double* a = &A[9 * k];
		Eigen::Matrix3d Am;
		Am << a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8];
		Eigen::JacobiSVD<Eigen::Matrix3d> svd(Am, Eigen::ComputeFullU | Eigen::ComputeFullV);
		Eigen::Matrix3d U = svd.matrixU();
		const Eigen::Matrix3d V = svd.matrixV();
		Eigen::Matrix3d R = U * V.transpose();
		if (det3(R) < 0.0) {   // 반사 방지: 가장 작은 특이값 축(열 2)을 뒤집는다
			U.col(2) *= -1.0;
			R = U * V.transpose();
		}
		bool finite = true;
		for (int r = 0; r < 3; ++r)
			for (int cc = 0; cc < 3; ++cc)
				if (!std::isfinite(R(r, cc))) finite = false;
		if (!finite) continue;
		float* P = &params[16 * k];
		for (int t = 0; t < 3; ++t) {
			P[t] = g_objC0[3 * k + t];
			P[3 + t] = (float)c[3 * k + t];
			for (int cc = 0; cc < 3; ++cc) P[6 + 3 * t + cc] = (float)R(t, cc);
		}
		P[15] = 1.0f;
	}
	cudaMemcpy(d_objParams, params.data(), sizeof(float) * 16 * K, cudaMemcpyHostToDevice);
	objectShapeMatchKernel << <blocks, threads >> > (
		N, d_pos_next, d_pos_rest, d_invMass, d_objId, d_objParams, g_objShapeStiffness);
}

// 위와 같은 계산을 디바이스에서만 한다. 호스트와 주고받는 것은 자기충돌이 켜졌을 때의 물체별 이동량(K개)뿐.
static void objectShapeMatchGPU(int N, int blocks, int threads)
{
	const int K = g_objNumComponents;
	const bool wOn = g_attActive && d_objW;       // 붙잡기 중에만 무게 있는 맞춤 (그 밖에는 예전 산술 그대로)
	const int stride = wOn ? 16 : 12;
	cudaMemsetAsync(d_objSums, 0, sizeof(double) * stride * K);
	objectShapeAccumKernel << <blocks, threads >> > (N, d_pos_next, d_objId, d_objRestRel, d_objShift, K, d_objSums,
		wOn ? d_objW : nullptr, stride);
	objectShapeSolveKernel << <(K + 63) / 64, 64 >> > (
		K, d_objSums, stride, d_objCountDev, d_objSumQ, d_objC0Dev, d_objShift, d_objParams, d_objDisp);

	// 자기충돌 활성 제한용 경계상자·이동량 (호스트 경로는 매 프레임 계산하지만 쓰는 곳은 자기충돌뿐)
	if (g_selfColEnabled && !g_squashActive) {
		if (K > g_objBoxCap) {
			if (d_objBoxes) cudaFree(d_objBoxes);
			cudaMalloc(&d_objBoxes, sizeof(float) * 6 * K);
			g_objBoxCap = K;
		}
		objectBoxesInitKernel << <(6 * K + 255) / 256, 256 >> > (K, d_objBoxes);
		objectBoxesKernel << <blocks, threads >> > (N, d_pos_next, d_objId, K, d_objBoxes);
		std::vector<float> disp(K);
		cudaMemcpy(disp.data(), d_objDisp, sizeof(float) * K, cudaMemcpyDeviceToHost);
		float maxDisp = 0.0f;
		for (int k = 0; k < K; ++k) maxDisp = fmaxf(maxDisp, disp[k]);
		g_objMaxBodyDisp = maxDisp;
		g_objBoxesValid = g_objShiftValid && std::isfinite(maxDisp);   // 첫 프레임은 이동량을 모르므로 제한하지 않는다
	}
	g_objShiftValid = true;

	objectShapeMatchKernel << <blocks, threads >> > (
		N, d_pos_next, d_pos_rest, d_invMass, d_objId, d_objParams, g_objShapeStiffness);
}

// ── Self-collision 호스트 단계 ───────────────────────────────────────────────
static void selfColFreeBuffers()
{
	if (d_scKeys) { cudaFree(d_scKeys); d_scKeys = nullptr; }
	if (d_scKeysSorted) { cudaFree(d_scKeysSorted); d_scKeysSorted = nullptr; }
	if (d_scIdx) { cudaFree(d_scIdx); d_scIdx = nullptr; }
	if (d_scIdxSorted) { cudaFree(d_scIdxSorted); d_scIdxSorted = nullptr; }
	if (d_scCellStart) { cudaFree(d_scCellStart); d_scCellStart = nullptr; }
	if (d_scCellEnd) { cudaFree(d_scCellEnd); d_scCellEnd = nullptr; }
	if (d_scContact) { cudaFree(d_scContact); d_scContact = nullptr; }
	if (d_scCount) { cudaFree(d_scCount); d_scCount = nullptr; }
	if (d_scDp) { cudaFree(d_scDp); d_scDp = nullptr; }
	if (d_scSortTemp) { cudaFree(d_scSortTemp); d_scSortTemp = nullptr; }
	if (d_scDisp2) { cudaFree(d_scDisp2); d_scDisp2 = nullptr; }
	if (d_scActive) { cudaFree(d_scActive); d_scActive = nullptr; }
	if (d_scDpCross) { cudaFree(d_scDpCross); d_scDpCross = nullptr; }
	if (d_scAccum) { cudaFree(d_scAccum); d_scAccum = nullptr; }
	if (d_scMaxOut) { cudaFree(d_scMaxOut); d_scMaxOut = nullptr; }
	if (d_scReduceTemp) { cudaFree(d_scReduceTemp); d_scReduceTemp = nullptr; }
	g_scSortTempBytes = 0;
	g_scReduceTempBytes = 0;
	g_selfColCap = 0;
}

static void selfColEnsureBuffers(int N)
{
	if (g_selfColCap == N && d_scKeys) return;
	selfColFreeBuffers();
	int table = 1024;
	while (table < 2 * N) table <<= 1;   // 칸 수 ≤ N 이므로 적재율 ≤ 0.5
	int bits = 0;
	while ((1 << bits) < table) ++bits;
	cudaMalloc(&d_scKeys, sizeof(unsigned int) * N);
	cudaMalloc(&d_scKeysSorted, sizeof(unsigned int) * N);
	cudaMalloc(&d_scIdx, sizeof(int) * N);
	cudaMalloc(&d_scIdxSorted, sizeof(int) * N);
	cudaMalloc(&d_scCellStart, sizeof(int) * table);
	cudaMalloc(&d_scCellEnd, sizeof(int) * table);
	cudaMalloc(&d_scContact, sizeof(int) * (size_t)N * SELFCOL_MAX_CONTACTS);
	cudaMalloc(&d_scCount, sizeof(int) * N);
	cudaMalloc(&d_scDp, sizeof(float3) * N);
	cudaMemset(d_scCount, 0, sizeof(int) * N);
	g_scSortTempBytes = 0;
	cub::DeviceRadixSort::SortPairs(nullptr, g_scSortTempBytes, d_scKeys, d_scKeysSorted, d_scIdx, d_scIdxSorted, N, 0, bits + 1);
	cudaMalloc(&d_scSortTemp, g_scSortTempBytes > 0 ? g_scSortTempBytes : 1);
	cudaMalloc(&d_scDisp2, sizeof(float) * N);
	cudaMalloc(&d_scActive, sizeof(unsigned char) * N);
	cudaMalloc(&d_scDpCross, sizeof(float3) * N);
	cudaMalloc(&d_scAccum, sizeof(float3) * N);
	cudaMalloc(&d_scMaxOut, sizeof(float));
	g_scReduceTempBytes = 0;
	cub::DeviceReduce::Max(nullptr, g_scReduceTempBytes, d_scDisp2, d_scMaxOut, N);
	cudaMalloc(&d_scReduceTemp, g_scReduceTempBytes > 0 ? g_scReduceTempBytes : 1);
	g_selfColCap = N;
	g_selfColTable = table;
	g_selfColBits = bits;
	printf("[SelfCol] buffers: N=%d hash table=%d cap=%d/particle (%.1f MB)\n", N, table, SELFCOL_MAX_CONTACTS,
		(sizeof(int) * (double)N * (6 + SELFCOL_MAX_CONTACTS) + sizeof(int) * 2.0 * table + sizeof(float3) * (double)N + g_scSortTempBytes) / 1048576.0);
}

// 간격 = 입자별 '가장 가까운 그래프 이웃' rest 거리의 중앙값. 그래프가 바뀌면(버퍼 주소 변경) 다시 잰다.
static void selfColEnsureSpacing(int N)
{
	if (g_selfColSpacingN == N && g_selfColSpacingSrc == d_nbrDist && g_selfColSpacing > 0.0f) return;
	if (!d_nbrDist || !d_offset || !d_nbrCount || cm_num_neighbors <= 0) return;
	std::vector<int> off(N), cnt(N);
	std::vector<float> dist(cm_num_neighbors);
	cudaMemcpy(off.data(), d_offset, sizeof(int) * N, cudaMemcpyDeviceToHost);
	cudaMemcpy(cnt.data(), d_nbrCount, sizeof(int) * N, cudaMemcpyDeviceToHost);
	cudaMemcpy(dist.data(), d_nbrDist, sizeof(float) * cm_num_neighbors, cudaMemcpyDeviceToHost);
	std::vector<float> mins;
	mins.reserve(N);
	for (int i = 0; i < N; ++i) {
		float m = FLT_MAX;
		for (int k = 0; k < cnt[i]; ++k) {
			const int e = off[i] + k;
			if (e < 0 || e >= cm_num_neighbors) continue;
			const float d = dist[e];
			if (d > 0.0f && isfinite(d) && d < m) m = d;
		}
		if (m < FLT_MAX) mins.push_back(m);
	}
	if (mins.empty()) return;
	std::nth_element(mins.begin(), mins.begin() + mins.size() / 2, mins.end());
	g_selfColSpacing = mins[mins.size() / 2];
	g_selfColSpacingN = N;
	g_selfColSpacingSrc = d_nbrDist;
	printf("[SelfCol] spacing (median nearest graph edge) = %.5g\n", g_selfColSpacing);
}

// 프레임당 1회: 최대 변위 → 해시 키(시작 위치) → 기수 정렬 → 칸 구간 → 입자별 후보 목록.
// 후보 수를 읽어 반복 단계 생략 여부를 정한다.
static void selfColBuild(int N, int blocks, int threads)
{
	selfColEnsureSpacing(N);
	const float dc = g_selfColRadiusScale * g_selfColSpacing;
	if (!(dc > 0.0f)) { g_selfColContacts = 0; return; }
	selfColEnsureBuffers(N);
	const float search = g_selfColSearchScale * dc;
	const float excl = g_selfColExcludeScale * dc;

	const unsigned int zero3[3] = { 0u, 0u, 0u };
	cudaMemcpyToSymbol(g_selfColCtr, zero3, sizeof(zero3));
	objectEnsureComponents(N);   // 물체 id (다른 물체 쌍의 규칙·제외 기준을 따로 쓰기 위해)

	// 활성 제한: 같은 물체 안 접촉을 끈 경우, 다른 물체 경계상자 근처 입자만 격자에 넣는다.
	// 경계상자는 objectShapeMatch 가 이번 프레임에 채운 것만 쓴다 (없으면 제한하지 않는다).
	const int K = g_objNumComponents;
	const bool useActive = !g_selfColWithinBody && g_objBoxesValid && d_objBoxes && d_objId && K >= 1 && K <= 64;
	const unsigned char* active = useActive ? d_scActive : nullptr;
	if (useActive) {
		const float inflate = search + 2.0f * g_objMaxBodyDisp + 2.0f * dc;
		selfColActiveKernel << <blocks, threads >> > (N, d_pos_curr, d_pos_next, d_objId, d_objBoxes, K, inflate, d_scActive);
	}
	g_objBoxesValid = false;

	// 두 점이 이번 프레임에 가까워질 수 있는 양 ≤ 2·max|p − x| (활성 입자만) → 격자 칸 = 탐색반경 + 그 값.
	// 너무 빠르면 비용이 폭증하므로 상한을 두고, 상한에 걸린 프레임을 센다(그 이상 빠른 접근은 관통할 수 있다).
	selfColDispKernel << <blocks, threads >> > (N, d_pos_curr, d_pos_next, active, d_scDisp2);
	cub::DeviceReduce::Max(d_scReduceTemp, g_scReduceTempBytes, d_scDisp2, d_scMaxOut, N);
	float maxDisp2 = 0.0f;
	cudaMemcpy(&maxDisp2, d_scMaxOut, sizeof(float), cudaMemcpyDeviceToHost);
	const float maxDisp = (isfinite(maxDisp2) && maxDisp2 > 0.0f) ? sqrtf(maxDisp2) : 0.0f;
	float reach = search + 2.0f * maxDisp;
	const float reachCap = search + 16.0f * dc;
	if (reach > reachCap) { reach = reachCap; ++g_selfColReachClamped; }
	g_selfColReach = reach;
	const float invCell = 1.0f / reach;
	const unsigned int mask = (unsigned int)g_selfColTable - 1u;
	const unsigned int sentinel = (unsigned int)g_selfColTable;      // mask 보다 크고 bits+1 비트 안

	selfColKeyKernel << <blocks, threads >> > (N, d_pos_curr, active, d_scKeys, d_scIdx, invCell, mask, sentinel);
	cub::DeviceRadixSort::SortPairs(d_scSortTemp, g_scSortTempBytes, d_scKeys, d_scKeysSorted, d_scIdx, d_scIdxSorted, N, 0, g_selfColBits + 1);
	cudaMemset(d_scCellStart, 0xFF, sizeof(int) * g_selfColTable);   // -1
	selfColCellRangeKernel << <blocks, threads >> > (N, d_scKeysSorted, d_scCellStart, d_scCellEnd, mask);
	selfColBuildKernel << <blocks, threads >> > (
		N, d_pos_curr, d_pos_next, d_pos_rest, d_scIdxSorted, d_scCellStart, d_scCellEnd,
		invCell, mask, search, excl * excl, d_objId, dc * dc, g_selfColFastRel * dc,
		active, g_selfColWithinBody ? 1 : 0, d_scContact, d_scCount);
	cudaDeviceSynchronize();
	unsigned int c3[3] = { 0u, 0u, 0u };
	cudaMemcpyFromSymbol(c3, g_selfColCtr, sizeof(c3));
	g_selfColContacts = (int)c3[0];
	g_selfColOverflow = (int)c3[1];
	g_selfColActive = useActive ? (int)c3[2] : N;
}

// ── 접촉 평균 강체 이동 (물체끼리 서서히 스며드는 문제) ──────────────────────
// Keep object shape 는 무게중심 기준이라, 닫힌 껍질 쿠션처럼 접촉점 비율 f 가 작으면 충돌 보정이 f 만큼 희석되어
// 다음 프레임에 되돌려진다 → 평형 침투 ≈ g·dt²/f 로 얇은 막을 서서히 뚫는다 (pillow 뷰어 증상).
// 수정: 보정 받은 점들의 평균 보정을 같은 물체의 나머지 점에도 ×s 만큼 준다 (단단한 물체일수록 통째로 밀린다).
// CPU 복제 seep_experiment.py (두 겹 쿠션 껍질·접촉점 약 3%, s=0.94, 150프레임):
//   이 이동 적용 → 바닥 간격 +1.05·d_c 유지(최소 +0.80), 떨림 0.007~0.06 d_c/f
//   물체 shape matching 을 반복마다 → +0.67·d_c (최소 +0.43), 떨림 0.16~0.22
static void objectContactPush(int N, int blocks, int threads, float3* pos)
{
	const int K = g_objNumComponents;
	if (K <= 0 || !d_objId || !d_scAccum || g_objShapeStiffness <= 0.0f) return;
	if (K > g_objPushCap) {
		if (d_objPushSum) cudaFree(d_objPushSum);
		if (d_objPushT) cudaFree(d_objPushT);
		cudaMalloc(&d_objPushSum, sizeof(float) * 4 * K);
		cudaMalloc(&d_objPushT, sizeof(float) * 4 * K);
		g_objPushCap = K;
	}
	cudaMemset(d_objPushSum, 0, sizeof(float) * 4 * K);
	objectPushSumKernel << <blocks, threads >> > (N, d_objId, d_scAccum, d_objPushSum);
	cudaDeviceSynchronize();
	std::vector<float> sums(4 * (size_t)K), T(4 * (size_t)K, 0.0f);
	cudaMemcpy(sums.data(), d_objPushSum, sizeof(float) * 4 * K, cudaMemcpyDeviceToHost);
	bool any = false;
	for (int k = 0; k < K; ++k) {
		const float n = sums[4 * k + 3];
		if (!(n > 0.0f)) continue;
		const float sc = g_objShapeStiffness / n;
		T[4 * k] = sums[4 * k] * sc;
		T[4 * k + 1] = sums[4 * k + 1] * sc;
		T[4 * k + 2] = sums[4 * k + 2] * sc;
		if (!isfinite(T[4 * k]) || !isfinite(T[4 * k + 1]) || !isfinite(T[4 * k + 2])) continue;
		T[4 * k + 3] = 1.0f;
		any = true;
	}
	if (!any) return;
	cudaMemcpy(d_objPushT, T.data(), sizeof(float) * 4 * K, cudaMemcpyHostToDevice);
	objectPushApplyKernel << <blocks, threads >> > (N, d_objId, d_invMass, d_scAccum, d_objPushT, pos);
	++g_objPushFrames;
}

// Squash/Press 체크포인트 형상 지표 — J 가 못 보는 두 가지를 같이 찍는다 (체크포인트 4번만 호출, 호스트 계산).
//   mid-band radial RMS : rest 기준 하중축 가운데 20% 띠 입자들의, 축에 수직인 반경 RMS (rest 대비 배율) → 옆으로 부푼 정도
//   edge stretch        : 그래프 엣지 중 현재 길이가 rest 의 1.5배 / 2배를 넘는 비율 → 입자가 이웃과 따로 튀어나간 정도
static inline float squashAxisValue(const float3& p, int ax) { return (ax == 0) ? p.x : (ax == 1) ? p.y : p.z; }

// 그래프 엣지 늘어남 집계: 현재 길이가 rest 의 1.5배 / 2배를 넘는 엣지 수와 rest 엣지 평균 길이.
static void countStretchedEdges(int N, const std::vector<float3>& cur, const std::vector<float3>& rest,
	long long& edges, long long& s15, long long& s20, double& meanRest)
{
	edges = 0; s15 = 0; s20 = 0; meanRest = 0.0;
	if (!(d_offset && d_nbrCount && d_nbrIdx && cm_num_neighbors > 0)) return;
	std::vector<int> off(N), cnt(N), nb(cm_num_neighbors);
	cudaMemcpy(off.data(), d_offset, sizeof(int) * N, cudaMemcpyDeviceToHost);
	cudaMemcpy(cnt.data(), d_nbrCount, sizeof(int) * N, cudaMemcpyDeviceToHost);
	cudaMemcpy(nb.data(), d_nbrIdx, sizeof(int) * cm_num_neighbors, cudaMemcpyDeviceToHost);
	for (int i = 0; i < N; ++i) {
		for (int k = 0; k < cnt[i]; ++k) {
			const int e = off[i] + k;
			if (e < 0 || e >= cm_num_neighbors) break;
			const int j = nb[e];
			if (j <= i || j >= N) continue;   // 무향 엣지를 한 번만
			const double rx = rest[i].x - rest[j].x, ry = rest[i].y - rest[j].y, rz = rest[i].z - rest[j].z;
			const double L0 = sqrt(rx * rx + ry * ry + rz * rz);
			if (!(L0 > 1e-12)) continue;
			const double qx = cur[i].x - cur[j].x, qy = cur[i].y - cur[j].y, qz = cur[i].z - cur[j].z;
			const double L = sqrt(qx * qx + qy * qy + qz * qz);
			++edges;
			meanRest += L0;
			if (!(L <= 1.5 * L0)) ++s15;   // NaN 도 튀어나간 것으로 센다
			if (!(L <= 2.0 * L0)) ++s20;
		}
	}
	if (edges > 0) meanRest /= (double)edges;
}

static void logSquashShapeMetrics(int N, const char* tag)
{
	if (!d_pos_curr || !d_pos_rest || N <= 0) return;
	std::vector<float3> cur(N), rest(N);
	cudaMemcpy(cur.data(), d_pos_curr, sizeof(float3) * N, cudaMemcpyDeviceToHost);
	cudaMemcpy(rest.data(), d_pos_rest, sizeof(float3) * N, cudaMemcpyDeviceToHost);
	const int ax = g_squashAxis;

	float lo = FLT_MAX, hi = -FLT_MAX;
	for (int i = 0; i < N; ++i) { const float v = squashAxisValue(rest[i], ax); lo = fminf(lo, v); hi = fmaxf(hi, v); }
	const float mid = 0.5f * (lo + hi), half = 0.1f * (hi - lo);
	std::vector<int> band;
	band.reserve(N / 4);
	double cr[3] = { 0.0, 0.0, 0.0 }, cc[3] = { 0.0, 0.0, 0.0 };
	for (int i = 0; i < N; ++i) {
		if (fabsf(squashAxisValue(rest[i], ax) - mid) > half) continue;
		if (!isfinite(cur[i].x) || !isfinite(cur[i].y) || !isfinite(cur[i].z)) continue;
		band.push_back(i);
		cr[0] += rest[i].x; cr[1] += rest[i].y; cr[2] += rest[i].z;
		cc[0] += cur[i].x; cc[1] += cur[i].y; cc[2] += cur[i].z;
	}
	double radial = 0.0;
	if (!band.empty()) {
		const double inv = 1.0 / (double)band.size();
		for (int a = 0; a < 3; ++a) { cr[a] *= inv; cc[a] *= inv; }
		double sr = 0.0, sc = 0.0;
		for (int i : band) {
			double dr[3] = { rest[i].x - cr[0], rest[i].y - cr[1], rest[i].z - cr[2] };
			double dcv[3] = { cur[i].x - cc[0], cur[i].y - cc[1], cur[i].z - cc[2] };
			dr[ax] = 0.0; dcv[ax] = 0.0;   // 축에 수직인 성분만
			sr += dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
			sc += dcv[0] * dcv[0] + dcv[1] * dcv[1] + dcv[2] * dcv[2];
		}
		radial = (sr > 0.0) ? sqrt(sc / sr) : 0.0;
	}

	long long edges = 0, s15 = 0, s20 = 0;
	double meanRest = 0.0;
	countStretchedEdges(N, cur, rest, edges, s15, s20, meanRest);
	const double e = (edges > 0) ? 100.0 / (double)edges : 0.0;
	printf("[Squash] shape %s: mid-band radial RMS x%.4f (n=%d) | edges >1.5x %.3f%%  >2x %.3f%% (of %lld)\n",
		tag, radial, (int)band.size(), s15 * e, s20 * e, edges);
}

// Ground 데모 충돌 과도응답 지표 (Launch 뒤, 호스트 계산). 모두 바닥 법선 n 기준, rest 대비 배율.
//   COM above floor : 무게중심의 바닥 위 높이 (rest 높이 단위)
//   height / lateral: n 방향 두께, n 에 수직인 반경 RMS → 눌림과 옆으로 부푼 정도
//   edges           : rest 의 1.5배 / 2배를 넘게 늘어난 엣지 비율 → 입자가 이웃과 따로 튀어나간 정도
//   |v - vcom|      : 무게중심 속도를 뺀 입자 속도 (p99 / 최대, rest 엣지 길이 / 프레임) → 따로 튀는 속도
//   ※ Throw 는 회전(ω×r)이 |v - vcom| 에 섞이므로 비교는 Drop 으로 한다.
static void logImpactMetrics(int N, int frame, float dt)
{
	if (!d_pos_curr || !d_pos_rest || !d_vel || N <= 0) return;
	std::vector<float3> cur(N), rest(N), vel(N);
	cudaMemcpy(cur.data(), d_pos_curr, sizeof(float3) * N, cudaMemcpyDeviceToHost);
	cudaMemcpy(rest.data(), d_pos_rest, sizeof(float3) * N, cudaMemcpyDeviceToHost);
	cudaMemcpy(vel.data(), d_vel, sizeof(float3) * N, cudaMemcpyDeviceToHost);
	const double nx = g_groundN[0], ny = g_groundN[1], nz = g_groundN[2];
	const double plane = (double)g_groundHeight + (double)g_groundRadius;

	double cr[3] = { 0.0, 0.0, 0.0 }, cc[3] = { 0.0, 0.0, 0.0 }, vc[3] = { 0.0, 0.0, 0.0 };
	int cnt = 0;
	for (int i = 0; i < N; ++i) {
		if (!isfinite(cur[i].x) || !isfinite(cur[i].y) || !isfinite(cur[i].z)) continue;
		cr[0] += rest[i].x; cr[1] += rest[i].y; cr[2] += rest[i].z;
		cc[0] += cur[i].x; cc[1] += cur[i].y; cc[2] += cur[i].z;
		vc[0] += vel[i].x; vc[1] += vel[i].y; vc[2] += vel[i].z;
		++cnt;
	}
	if (cnt == 0) return;
	for (int a = 0; a < 3; ++a) { cr[a] /= cnt; cc[a] /= cnt; vc[a] /= cnt; }

	double hrLo = 1e300, hrHi = -1e300, hcLo = 1e300, hcHi = -1e300, latR = 0.0, latC = 0.0;
	int contacts = 0;
	std::vector<float> dv;
	dv.reserve(cnt);
	for (int i = 0; i < N; ++i) {
		if (!isfinite(cur[i].x) || !isfinite(cur[i].y) || !isfinite(cur[i].z)) continue;
		const double ax = rest[i].x - cr[0], ay = rest[i].y - cr[1], az = rest[i].z - cr[2];
		const double an = ax * nx + ay * ny + az * nz;
		hrLo = fmin(hrLo, an); hrHi = fmax(hrHi, an);
		latR += ax * ax + ay * ay + az * az - an * an;
		const double bx = cur[i].x - cc[0], by = cur[i].y - cc[1], bz = cur[i].z - cc[2];
		const double bn = bx * nx + by * ny + bz * nz;
		hcLo = fmin(hcLo, bn); hcHi = fmax(hcHi, bn);
		latC += bx * bx + by * by + bz * bz - bn * bn;
		if (cur[i].x * nx + cur[i].y * ny + cur[i].z * nz - plane <= g_groundSlop) ++contacts;
		const double ux = vel[i].x - vc[0], uy = vel[i].y - vc[1], uz = vel[i].z - vc[2];
		dv.push_back((float)sqrt(ux * ux + uy * uy + uz * uz));
	}
	const double restH = fmax(hrHi - hrLo, 1e-12);
	const double comGap = (cc[0] * nx + cc[1] * ny + cc[2] * nz - plane) / restH;
	const double lateral = (latR > 0.0) ? sqrt(latC / latR) : 0.0;

	long long edges = 0, s15 = 0, s20 = 0;
	double meanRest = 0.0;
	countStretchedEdges(N, cur, rest, edges, s15, s20, meanRest);
	const double e = (edges > 0) ? 100.0 / (double)edges : 0.0;

	float p99 = 0.0f, vmax = 0.0f;
	if (!dv.empty()) {
		const size_t k99 = (size_t)(0.99 * (dv.size() - 1));
		std::nth_element(dv.begin(), dv.begin() + k99, dv.end());
		p99 = dv[k99];
		vmax = *std::max_element(dv.begin() + k99, dv.end());
	}
	const double toEdge = (meanRest > 0.0) ? (double)dt / meanRest : 0.0;   // 속도 → rest 엣지 길이 / 프레임
	printf("[Impact] f%3d | COM above floor %.3f H | contacts %6d | height x%.3f  lateral x%.3f | edges >1.5x %.2f%%  >2x %.3f%% | |v-vcom| p99 %.2f  max %.2f edge/frame\n",
		frame, comGap, contacts, (hcHi - hcLo) / restH, lateral, s15 * e, s20 * e, p99 * toEdge, vmax * toEdge);
}

static void runXPBDSimulation(FORWARD::ChainMail& cm, int solverIters, float dt, float underRelax, float velDamping, float3 gravity, bool uploadFromCPU, bool downloadToCPU, const std::vector<int>* uploadIndices)
{
	//ensureChainmailGPU(cm);
	if (uploadFromCPU) {
		uploadChainmailPositions(cm);
	}
	else if (uploadIndices && !uploadIndices->empty()) {
		//uploadChainmailPositionsIndexed(cm, *uploadIndices);
	}

	const int N = static_cast<int>(cm.numElements());//처리할 총 가우시안 개수
	if (N <= 0) return;
	const int threads = 256;
	const int blocks = (N + threads - 1) / threads;//가우시안 1개당 스레드 1개

	if (g_useXPBDDistanceConstraint && cm_num_neighbors > 0) {
		cudaMemset(d_lambda_curr, 0, sizeof(float) * cm_num_neighbors);
		cudaMemset(d_lambda_next, 0, sizeof(float) * cm_num_neighbors);
	}
	// XPBD 규약: Lagrange multiplier는 매 substep 시작 시 0으로 리셋한다.
	if (g_useVolumeConstraint && d_lambda_vol) {
		cudaMemset(d_lambda_vol, 0, sizeof(float) * N);
	}
	if (g_useGaussianNH && d_lambda_gnhD && d_lambda_gnhH) {
		cudaMemset(d_lambda_gnhD, 0, sizeof(float) * N);
		cudaMemset(d_lambda_gnhH, 0, sizeof(float) * N);
	}
	const bool distGS = g_useXPBDDistanceConstraint && g_useDistanceGS && d_gsEdge && g_gsNumEdges > 0;
	const bool volGS = g_useVolumeConstraint && g_useVolumeGS && d_volGSOrder && g_volGSColorOffset.size() > 1;
	if (distGS) {
		cudaMemset(d_gsLambda, 0, sizeof(float) * g_gsNumEdges);
	}
	// 클램프 카운터 리셋 (이 프레임의 발동 횟수만 센다)
	if (g_volClampStats && g_useVolumeConstraint) {
		const unsigned int zero3[3] = { 0u, 0u, 0u };
		cudaMemcpyToSymbol(g_clampCtr, zero3, sizeof(zero3));
	}

	// Squash 시작 프레임에 한 번: 이 실험에서 무시되는 Ground 데모 설정과, 응답을 바꾸는 XPBD 설정을 콘솔에 남긴다.
	{
		static bool s_squashWasActive = false;
		if (g_squashActive && !s_squashWasActive) {
			if (g_objShapeStiffness > 0.0f || g_groundEnabled || g_groundGravity > 0.0f || g_selfColEnabled || g_groundPaused) {
				printf("[Squash] Ground demo settings ignored during squash: object shape %.3f | floor %s | gravity %.3g | self-collision %s | pause %s\n",
					g_objShapeStiffness, g_groundEnabled ? "on" : "off", g_groundGravity,
					g_selfColEnabled ? "on" : "off", g_groundPaused ? "on" : "off");
			}
			printf("[Squash] XPBD: vel damping %.3f/frame | shape matching %s (robust rotation %s) | distance %s | volume %s | GaussianNH %s\n",
				velDamping, g_useXPBDShapeMatching ? "on" : "off", g_xpbdShapeRobustPolar ? "on" : "off",
				g_useXPBDDistanceConstraint ? "on" : "off", g_useVolumeConstraint ? "on" : "off",
				g_useGaussianNH ? "on" : "off");
		}
		s_squashWasActive = g_squashActive;
	}

	// Squash 테스트: 슬랩 pin을 매 프레임 재적용.
	// applySeedCommands가 이미 invMass=1로 리셋해놓았으므로 여기서 다시 0을 박는다.
	if (g_squashActive && (g_squashPress || (g_squashTopCount > 0 && g_squashBotCount > 0))) {
		if (g_squashAutoLog && g_squashDwell > 0) {
			// 정착 구간: 변위를 고정한 채 솔버만 돌린다. 마지막 프레임에 기록 예약.
			if (--g_squashDwell == 0) { g_squashLogPending = true; ++g_squashCkptIdx; }
		}
		else {
			g_squashCurDisp = fminf(g_squashCurDisp + g_squashRampPerSec * dt, g_squashMaxDisp);
			// 체크포인트를 넘었으면 정확히 그 값으로 맞추고 정착에 들어간다.
			if (g_squashAutoLog && g_squashCkptIdx < 4) {
				const float target = g_squashCkpt[g_squashCkptIdx] * g_squashMaxDisp;
				if (g_squashCurDisp >= target) {
					g_squashCurDisp = target;
					g_squashDwell = g_squashDwellFrames;
				}
			}
		}
		const int nTotal = g_squashTopCount + g_squashBotCount;
		const int sqBlocks = (nTotal + threads - 1) / threads;
		if (!g_squashPress && nTotal > 0) applySquashSlabKernel << <sqBlocks, threads >> > (   // press 는 슬랩을 잡지 않는다
			g_squashTopCount, g_squashBotCount,
			d_squashTopIdx, d_squashBotIdx,
			d_squashTopRest, d_squashBotRest,
			d_pos_curr, d_vel, d_invMass,
			g_squashAxis, g_squashCurDisp, g_squashSlip ? 1 : 0);
	}

	// 1) Predict
	//관성에 의한 위치 예측(stiffness 를 따지기 전, 관성과, 중력에 의해 다음프레임 좌표를 1차로 예측하는 도화지(pos_pred)를 만드는 단계)
	// [추가된 지진 효과용 시간 추적 변수]
	static float accumulated_time = 0.0f;
	accumulated_time += dt;

	// [지진 파라미터]
	float quake_amplitude = 150.0f; // 흔들림 강도
	float quake_frequency = 30.0f;  // 흔들림 속도

	// const인 xpbdGravity(0,0,0)를 복사해서 새로운 가변 중력 벡터를 만듭니다.
	float3 dynamic_gravity = gravity;

	// X축(좌우)에만 사인파 진동을 더해줍니다. (기준점이 0이므로 순수하게 진동만 들어갑니다)
	//dynamic_gravity.x += quake_amplitude * sinf(quake_frequency * accumulated_time);

	// Object mode 1/2: 물체 단위 반발 충격량은 predict 전에 속도에 더한다.
	// Squash 실험(FEM 비교 하네스) 중에는 Ground 데모 기능을 적용하지 않는다 — 슬라이더 값은 두고 적용만 건너뛴다.
	// (예전에는 Prepare 로 켠 물체 형상 유지 0.05 가 압축 실험에도 걸려, 옆으로 부푸는 응답을 매 프레임 강체 rest 모양으로 되돌렸다.)
	const bool groundOn = g_groundEnabled && !g_squashActive;
	bool objShape = (g_objShapeStiffness > 0.0f) && d_pos_rest != nullptr && !g_squashActive;
	bool objContact = g_objBodyContact && groundOn && d_pos_rest != nullptr;
	std::chrono::high_resolution_clock::time_point objT0;
	if (objShape || objContact) {
		objT0 = std::chrono::high_resolution_clock::now();
		if (objectEnsureComponents(N)) {       // 물체 = 그래프 연결 성분
			objectEnsureRestCache(N);
			if (objContact) objectBodyContact(N, blocks, threads, dt);
		}
		else {
			objShape = false;
			objContact = false;
		}
	}
	xpbdPredictKernel << <blocks, threads >> > (
		N, d_pos_curr, 
		d_pos_next, d_vel, 
		d_invMass, 
		dt, 
		dynamic_gravity);

	// Object mode 2/2: 전역 shape matching은 predict 직후 1회 (반복마다 적용과 실험 결과 동일).
	if (objShape) {
		if (g_objShapeGPU && g_objNumComponents <= OBJ_GPU_MAX_K && d_objRestRel) objectShapeMatchGPU(N, blocks, threads);
		else objectShapeMatch(N, blocks, threads);
	}
	if (objShape || objContact) {
		g_objHostMsAccum += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - objT0).count();
		if (++g_objTimedFrames >= 100) {
			printf("[Object] host+transfer %.3f ms/frame (GPU sync 포함) | bodies %d | shape %.2f | body contact %s | impulses %d\n",
				g_objHostMsAccum / g_objTimedFrames, g_objNumComponents, g_objShapeStiffness, objContact ? "on" : "off", g_objImpulseCount);
			g_objHostMsAccum = 0.0;
			g_objTimedFrames = 0;
			g_objImpulseCount = 0;
		}
	}

	// 자기충돌 후보 탐색: predict(+물체 형상 유지) 직후 위치로 프레임당 1회.
	if (g_selfColEnabled && d_pos_rest && !g_squashActive) {
		const auto scT0 = std::chrono::high_resolution_clock::now();
		selfColBuild(N, blocks, threads);
		g_selfColBuildMsAccum += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - scT0).count();
		g_selfColContactsAccum += g_selfColContacts;
		if (++g_selfColTimedFrames >= 100) {
			printf("[SelfCol] build %.3f ms/frame (GPU sync 포함) | d_c %.4g (spacing %.4g) | active %d / %d | reach %.1f d_c (clamped %d frames) | candidates avg %.0f | cap overflow %d | body push %d frames | within-body %s\n",
				g_selfColBuildMsAccum / g_selfColTimedFrames, g_selfColRadiusScale * g_selfColSpacing, g_selfColSpacing,
				g_selfColActive, N,
				g_selfColReach / fmaxf(g_selfColRadiusScale * g_selfColSpacing, 1e-12f), g_selfColReachClamped,
				(double)g_selfColContactsAccum / g_selfColTimedFrames, g_selfColOverflow, g_objPushFrames,
				g_selfColWithinBody ? "on" : "off");
			g_selfColReachClamped = 0;
			g_objPushFrames = 0;
			g_selfColBuildMsAccum = 0.0;
			g_selfColContactsAccum = 0;
			g_selfColTimedFrames = 0;
		}
	}
	else {
		g_selfColContacts = 0;
	}
	if (g_selfColEnabled && g_selfColContacts > 0 && d_scAccum) {
		cudaMemset(d_scAccum, 0, sizeof(float3) * N);   // 이번 프레임 다른 물체 보정 누적
	}

	// 2) Solve distance constraints (Jacobi ping-pong for pos/lambda)
	float3* pred_in = d_pos_next;
	float3* pred_out = d_pos_xpbd_tmp;
	float* lambda_in = d_lambda_curr;
	float* lambda_out = d_lambda_next;

	//메인 솔버 예측된 위치(pos_pred_in)들을 놔두고 보니, 점들 사이의 원래 거리(L0)가 무너져 있음. 
	//이를 XPBD 수식으로 교정 -> pos_pred_out
	cudaEvent_t volStart, volMid, volEnd;
	cudaEventCreate(&volStart);
	cudaEventCreate(&volMid);
	cudaEventCreate(&volEnd);
	// ★ 이벤트는 '부피제약 ON + gather + 워프커널' 경로에서만 record 된다.
	//   그 밖의 경우 cudaEventElapsedTime이 실패하고 출력 변수를 건드리지 않아
	//   초기화 안 된 스택 쓰레기(예: 2.49e36 ms)가 찍혔다. 실제 기록 여부를 추적한다.
	bool volTimed = false;
	static int frame = 0;
	// 타원체 접촉: 바닥 법선 방향 반경을 스텝마다 한 번 (반복마다 입자당 4 B 만 읽게)
	const bool contactOn = (g_contactShapeOn || g_contactShapeGround) && d_contactM && g_contactN == N;
	if (contactOn && g_contactShapeGround && g_groundEnabled) {
		contactGroundRadiusKernel << <blocks, threads >> > (N, d_contactM,
			make_float3(g_groundN[0], g_groundN[1], g_groundN[2]), d_contactRg);
	}
	const float* cRg = (contactOn && g_contactShapeGround && g_groundEnabled) ? d_contactRg : nullptr;
	// 운동학 충돌체: 바뀐 자세만 올리고, 이번 스텝 통계를 비운다
	const bool kinColOn = (g_kinColCount > 0);
	if (kinColOn) {
		if (!d_kinCol) {
			cudaMalloc(&d_kinCol, sizeof(KinCollider) * MAX_KIN_COLLIDERS);
			cudaMalloc(&d_kinColHits, sizeof(int) * MAX_KIN_COLLIDERS);
			cudaMalloc(&d_kinColPush, sizeof(float) * KINCOL_ACC * MAX_KIN_COLLIDERS);
			g_kinColDirty = true;
		}
		if (g_kinColDirty) {
			cudaMemcpy(d_kinCol, g_kinCol, sizeof(KinCollider) * g_kinColCount, cudaMemcpyHostToDevice);
			g_kinColDirty = false;
		}
		cudaMemsetAsync(d_kinColHits, 0, sizeof(int) * g_kinColCount);
		cudaMemsetAsync(d_kinColPush, 0, sizeof(float) * KINCOL_ACC * g_kinColCount);
	}
	static cudaEvent_t loopT0 = nullptr, loopT1 = nullptr;
	if (g_physTiming) {
		if (!loopT0) { cudaEventCreate(&loopT0); cudaEventCreate(&loopT1); }
		cudaEventRecord(loopT0);
	}
	for (int it = 0; it < solverIters; ++it) {
		if (distGS) {
			// 색마다 커널 1번, pred_in 을 제자리 갱신 (swap 없음). 과소이완은 적용하지 않는다.
			const int nColors = int(g_gsColorOffset.size()) - 1;
			for (int c = 0; c < nColors; ++c) {
				const int begin = g_gsColorOffset[c];
				const int count = g_gsColorOffset[c + 1] - begin;
				if (count <= 0) continue;
				xpbdDistanceGSColorKernel << <(count + threads - 1) / threads, threads >> > (
					begin, count, N, pred_in, d_invMass,
					d_gsEdge, d_gsRest, d_gsStiff, d_gsLambda,
					dt, g_xpbdInvMassScale, g_xpbdStiffnessScale, g_xpbdDistanceCompliance);
			}
		}
		else if (g_useXPBDDistanceConstraint) {
			xpbdSolveJacobiKernel << <blocks, threads >> > (
				N,
				pred_in,
				pred_out,
				d_invMass,
				d_offset,
				d_nbrCount,
				d_nbrIdx,
				d_nbrDist,
				d_nbrStiff,
				lambda_in,
				lambda_out,
				dt,
				g_xpbdInvMassScale,
				g_xpbdStiffnessScale,
				underRelax,
				g_xpbdDistanceCompliance);
			std::swap(pred_in, pred_out);
			std::swap(lambda_in, lambda_out);
		}

		// Independent A/B path: Gaussian-cluster Stable Neo-Hookean block XPBD.
		// It neither calls nor writes any legacy volume/shape-matching kernel state.
		if (g_useGaussianNH && d_volOffset && d_volRevOffset && d_gnhValid &&
			d_gnhRestC && d_gnhRestSinv && d_gnhClusterF &&
			d_gnhClusterCoefD && d_gnhClusterCoefH) {
			xpbdGaussianNHClusterSolveKernel << <blocks, threads >> > (
				N, pred_in, d_pos_rest, d_invMass,
				d_volOffset, d_volCount, d_volIdx,
				d_gnhRestC, d_gnhRestSinv, d_gnhValid, d_V_rest,
				d_lambda_gnhD, d_lambda_gnhH,
				g_gnhYoung, g_gnhPoisson, g_gnhComplianceScale,
				dt, g_xpbdInvMassScale,
				d_gnhClusterF, d_gnhClusterCoefD, d_gnhClusterCoefH);
			xpbdGaussianNHGatherApplyKernel << <blocks, threads >> > (
				N, pred_in, pred_out, d_pos_rest, d_invMass,
				d_volRevOffset, d_volRevCount, d_volRevIdx,
				d_gnhRestC, d_gnhRestSinv, d_gnhValid,
				d_gnhClusterF, d_gnhClusterCoefD, d_gnhClusterCoefH,
				d_volRestLen, g_xpbdInvMassScale, underRelax);
			std::swap(pred_in, pred_out);
		}

		// 부피 제약(정수압)은 shape matching(편차)보다 '먼저' 푼다.
		// Macklin & Muller (2021): 부피를 먼저 회복한 뒤 모양을 다듬는 순서가 수렴에 유리하다.
		if (g_useVolumeConstraint && d_volOffset && d_vol_dp_sum) {
			if (volGS) {
				// 색마다 커널 1번, pred_in 제자리 갱신 (swap 없음). 클러스터당 1스레드.
				const int nColors = int(g_volGSColorOffset.size()) - 1;
				for (int c = 0; c < nColors; ++c) {
					const int begin = g_volGSColorOffset[c];
					const int count = g_volGSColorOffset[c + 1] - begin;
					if (count <= 0) continue;
					xpbdVolumeGSColorKernel << <(count + 63) / 64, 64 >> > (
						begin, count, d_volGSOrder, N, pred_in, d_invMass,
						d_volOffset, d_volCount, d_volIdx,
						d_detSigmaRest, d_matType, d_lambda_vol,
						g_volUsePhysicalAlpha ? d_alpha_vol : nullptr,
						g_volCompliance, dt, g_xpbdInvMassScale, d_volRestLen);
				}
			}
			else if (g_volGatherMode == 1 && d_volRevOffset && d_volClusterCoef) {
				// gather 모드: 클러스터 solve(칠판 쓰기) → 각자 읽어서 적용. atomic 없음.
				if (volUseWarpKernel()) {
					// 클러스터당 1워프. 블록 256스레드 = 8워프 = 8클러스터. 처리가능
					const int WPB = threads >> 5;
					const int gridW = (N + WPB - 1) / WPB;
					cudaEventRecord(volStart);
					if (volUseWarpOnePassKernel()) {
						xpbdVolumeClusterSolveWarpOnePassKernel << <gridW, threads >> > (
							N, pred_in, d_invMass,
							d_volOffset, d_volCount, d_volIdx,
							d_detSigmaRest, d_matType, d_lambda_vol,
							g_volUsePhysicalAlpha ? d_alpha_vol : nullptr,
							g_volCompliance, dt, g_xpbdInvMassScale,
							d_volClusterC, d_volClusterSinv, d_volClusterCoef);
					}
					else {
						xpbdVolumeClusterSolveWarpKernel << <gridW, threads >> > (
							N, pred_in, d_invMass,
							d_volOffset, d_volCount, d_volIdx,
							d_detSigmaRest, d_matType, d_lambda_vol,
							g_volUsePhysicalAlpha ? d_alpha_vol : nullptr,
							g_volCompliance, dt, g_xpbdInvMassScale,
							d_volClusterC, d_volClusterSinv, d_volClusterCoef);
					}
					cudaEventRecord(volMid);
				}
				else
				xpbdVolumeClusterSolveKernel << <blocks, threads >> > (
					N,
					pred_in,
					d_invMass,
					d_volOffset,
					d_volCount,
					d_volIdx,
					d_detSigmaRest,
					d_matType,
					d_lambda_vol,
					g_volUsePhysicalAlpha ? d_alpha_vol : nullptr, // Step 3: 물성 앵커 α(i)
					g_volCompliance,
					dt,
					g_xpbdInvMassScale,
					d_volClusterC,
					d_volClusterSinv,
					d_volClusterCoef);
				if (volUseWarpKernel()) {
					// 입자당 1워프. h_j 편차(0~937)로 인한 워프 내 부하 불균형을 없앤다.
					const int WPB = threads >> 5;
					const int gridW = (N + WPB - 1) / WPB;
					xpbdVolumeGatherApplyWarpKernel << <gridW, threads >> > (
						N, pred_in, pred_out, d_invMass,
						d_volRevOffset, d_volRevCount, d_volRevIdx,
						d_volClusterC, d_volClusterSinv, d_volClusterCoef,
						d_volRestLen, g_xpbdInvMassScale, underRelax);
					cudaEventRecord(volEnd);
					cudaEventSynchronize(volEnd);
					volTimed = true;   // 세 이벤트 모두 record 됨 → 타이밍 유효
				}
				else {
					xpbdVolumeGatherApplyKernel << <blocks, threads >> > (
						N,
						pred_in,
						pred_out,
						d_invMass,
						d_volRevOffset,
						d_volRevCount,
						d_volRevIdx,
						d_volClusterC,
						d_volClusterSinv,
						d_volClusterCoef,
						d_volRestLen,
						g_xpbdInvMassScale,
						underRelax);
				}
			}
			else {
				// scatter 모드: 전 멤버 산란(accumulate) → 평균 적용(apply).
				cudaMemset(d_vol_dp_sum, 0, sizeof(float3) * N);
				cudaMemset(d_vol_dp_count, 0, sizeof(int) * N);
				xpbdVolumeAccumulateKernel << <blocks, threads >> > (
					N,
					pred_in,
					d_invMass,
					d_volOffset,   // k-ring 볼륨 클러스터 CSR (k=1이면 그래프 이웃과 동일)
					d_volCount,
					d_volIdx,
					d_detSigmaRest,
					d_matType,
					d_lambda_vol,
					g_volUsePhysicalAlpha ? d_alpha_vol : nullptr, // Step 3: 물성 앵커 α(i). nullptr이면 손 튜닝 상수
					g_volCompliance,
					dt,
					g_xpbdInvMassScale,
					d_vol_dp_sum,
					d_vol_dp_count);
				xpbdVolumeApplyKernel << <blocks, threads >> > (
					N,
					pred_in,
					pred_out,
					d_invMass,
					d_vol_dp_sum,
					d_vol_dp_count,
					d_volRestLen,
					g_xpbdInvMassScale,
					underRelax);
			}
			if (!volGS) std::swap(pred_in, pred_out);
		}

		if (g_useRegionBalloon && g_regionBalloonCount >= 4 && g_regionBalloonDetRest > 1e-30f) {
			const int regionBlocks = (g_regionBalloonCount + threads - 1) / threads;
			regionBalloonClearStatsKernel << <1, 32 >> > (d_regionBalloonStats);
			regionBalloonSumKernel << <regionBlocks, threads >> > (
				g_regionBalloonCount,
				d_regionBalloonIdx,
				pred_in,
				d_regionBalloonStats);
			regionBalloonCovKernel << <regionBlocks, threads >> > (
				g_regionBalloonCount,
				d_regionBalloonIdx,
				pred_in,
				d_regionBalloonStats);

			// The region kernel writes only selected nodes, so keep non-region particles intact.
			cudaMemcpy(pred_out, pred_in, sizeof(float3) * N, cudaMemcpyDeviceToDevice);
			regionBalloonApplyKernel << <regionBlocks, threads >> > (
				g_regionBalloonCount,
				d_regionBalloonIdx,
				pred_in,
				pred_out,
				d_invMass,
				d_regionBalloonStats,
				g_regionBalloonDetRest,
				g_regionBalloonRestScale,
				g_regionBalloonCompliance,
				g_regionBalloonStrength,
				g_regionBalloonMaxStep,
				dt,
				g_xpbdInvMassScale,
				underRelax);
			std::swap(pred_in, pred_out);
		}

		if (g_useXPBDShapeMatching) {
			xpbdShapeMatchingKernel << <blocks, threads >> > (
				N,
				pred_in,
				pred_out,
				d_pos_rest,
				d_invMass,
				d_offset,
				d_nbrCount,
				d_nbrIdx,
				dt,
				g_xpbdInvMassScale,
				g_xpbdShapeCompliance,
				g_xpbdShapeBlend,
				g_xpbdShapeRobustPolar ? 1 : 0);
			std::swap(pred_in, pred_out);
		}

		if (g_useXPBDAngleConstraint) {
			// Two-pass Jacobi:
			// 1) symmetric delta accumulation to p0/p1/p2 using atomic adds
			// 2) averaged apply per particle
			cudaMemset(d_angle_dp_sum, 0, sizeof(float3) * N);
			cudaMemset(d_angle_dp_count, 0, sizeof(int) * N);

			xpbdAngleAccumulateKernel << <blocks, threads >> > (
				N,
				pred_in,
				d_invMass,
				d_offset,
				d_nbrCount,
				d_nbrIdx,
				d_pair_next_idx,
				d_rest_cos,
				d_angle_dp_sum,
				d_angle_dp_count,
				dt,
				g_xpbdAngleCompliance,
				g_xpbdInvMassScale);
			xpbdAngleApplyKernel << <blocks, threads >> > (
				N,
				pred_in,
				pred_out,
				d_invMass,
				d_angle_dp_sum,
				d_angle_dp_count,
				g_xpbdAngleBlend);
			std::swap(pred_in, pred_out);
		}

		// 미끄럼 평판이면 매 반복 하중축 변위를 되돌린다 (접선 두 축은 자유).
		if (g_squashActive && !g_squashPress && g_squashSlip && g_squashTopCount > 0 && g_squashBotCount > 0) {
			const int nSlab = g_squashTopCount + g_squashBotCount;
			pinSquashSlabAxisKernel << <(nSlab + threads - 1) / threads, threads >> > (
				g_squashTopCount, g_squashBotCount,
				d_squashTopIdx, d_squashBotIdx,
				d_squashTopRest, d_squashBotRest,
				pred_in, g_squashAxis, g_squashCurDisp);
		}

		// Press 모드: 위·아래 평판을 바닥과 같은 규칙으로 건다 (슬랩 핀과 같은 자리 — 두 경계조건만 다르게 A/B).
		if (g_squashActive && g_squashPress) {
			squashPressProjectKernel << <blocks, threads >> > (
				N, pred_in, d_invMass, g_squashAxis,
				g_squashLo + (g_squashPressFromLo ? g_squashCurDisp : 0.0f),
				g_squashHi - (g_squashPressFromLo ? 0.0f : g_squashCurDisp));
		}

		// 자기충돌: 읽기(보정 계산) → 쓰기(적용) 2단계. 후보가 없는 프레임은 커널을 띄우지 않는다.
		if (g_selfColEnabled && g_selfColContacts > 0) {
			selfColSolveKernel << <blocks, threads >> > (
				N, pred_in, d_pos_curr, d_pos_rest, d_invMass, d_scContact, d_scCount,
				g_selfColRadiusScale * g_selfColSpacing, g_xpbdInvMassScale, d_scDp, d_scDpCross);
			selfColApplyKernel << <blocks, threads >> > (N, pred_in, d_scCount, d_scDp, d_scDpCross, d_scAccum);
		}

		// 운동학 충돌체 (바닥 직전): 제약이 충돌체 안으로 끌어넣은 것을 되돌린다.
		if (kinColOn) {
			kinColliderProjectKernel << <blocks, threads >> > (
				N, pred_in, d_pos_curr, d_invMass, d_kinCol, g_kinColCount, g_kinColMargin, g_kinColFriction,
				d_kinColHits, d_kinColPush, (it == solverIters - 1) ? 1 : 0,
				(contactOn && g_contactShapeOn) ? d_contactM : nullptr, (contactOn && g_contactShapeOn) ? d_contactRmax : nullptr);
		}

		// 바닥 접촉은 반복의 '마지막' 연산이다 — 이번 반복의 다른 제약이 바닥 아래로 끌어내린 것을 되돌린다.
		if (groundOn) {
			groundProjectKernel << <blocks, threads >> > (
				N, pred_in, d_invMass,
				make_float3(g_groundN[0], g_groundN[1], g_groundN[2]),
				g_groundHeight + g_groundRadius, cRg);
		}
	}
	if (g_physTiming && loopT0) {
		cudaEventRecord(loopT1);
		cudaEventSynchronize(loopT1);
		float loopMs = 0.0f;
		if (cudaEventElapsedTime(&loopMs, loopT0, loopT1) == cudaSuccess) {
			g_physTimingSum += loopMs;
			++g_physTimingCount;
		}
	}
	frame++;
	if (frame % 100 == 0)
	{
		// volTimed 가 false면 이벤트가 record 되지 않아 측정값이 없다.
		// (예전엔 그대로 초기화 안 된 변수를 찍어 2.49e36 ms 같은 쓰레기가 나왔다.)
		if (volTimed) {
			float solve_ms = 0.0f, gather_ms = 0.0f, total_ms = 0.0f;
			if (cudaEventElapsedTime(&solve_ms, volStart, volMid) != cudaSuccess) solve_ms = 0.0f;
			if (cudaEventElapsedTime(&gather_ms, volMid, volEnd) != cudaSuccess) gather_ms = 0.0f;
			if (cudaEventElapsedTime(&total_ms, volStart, volEnd) != cudaSuccess) total_ms = 0.0f;
			printf("[VolKernel] Solve : %.6f ms | Gather : %.6f ms | Total : %.6f ms\n",
				solve_ms, gather_ms, total_ms);
		}
	}
	cudaEventDestroy(volStart);
	cudaEventDestroy(volMid);
	cudaEventDestroy(volEnd);
	// 물체끼리 접촉한 프레임: 보정 받은 점들의 평균 보정을 같은 물체 나머지 점에도 준다 (스며듦 방지), 이후 바닥 재투영.
	if (g_selfColEnabled && g_selfColContacts > 0 && objShape) {
		objectContactPush(N, blocks, threads, pred_in);
		if (g_groundEnabled) {
			groundProjectKernel << <blocks, threads >> > (
				N, pred_in, d_invMass,
				make_float3(g_groundN[0], g_groundN[1], g_groundN[2]),
				g_groundHeight + g_groundRadius, cRg);
		}
	}

	// 3) Velocity update and commit x_{t+1}
	//최종 속도 및 위치 확정
	if (groundOn) {
		// 바닥이 켜져 있을 때만 접촉 응답(반발·마찰) 판. 닿지 않은 입자는 아래 기존 커널과 산술이 같다.
		xpbdUpdateVelocityGroundKernel << <blocks, threads >> > (
			N,
			d_pos_curr,
			pred_in,
			d_vel,
			d_pos_curr,
			d_invMass,
			dt,
			velDamping,
			make_float3(g_groundN[0], g_groundN[1], g_groundN[2]),
			g_groundHeight + g_groundRadius,
			g_groundSlop,
			g_groundFriction,
			g_objBodyContact ? 0.0f : g_groundRestitution,   // 물체 단위 반발이 켜지면 입자 단위 반발은 끈다 (이중 적용 + e와 무관한 튕김)
			2.0f * g_groundGravity * dt, cRg);
	}
	else {
		xpbdUpdateVelocityKernel << <blocks, threads >> > (
			N,
			d_pos_curr,
			pred_in,
			d_vel,
			d_pos_curr,
			d_invMass,
			dt,
			velDamping);
	}

	g_lastActiveCount = N;
	g_lastActiveRatio = (N > 0) ? 1.0f : 0.0f;

	// Ground 데모 충돌 관찰: Launch 뒤 정해진 프레임 동안 3프레임마다 과도응답 지표를 찍는다.
	if (g_impactLogLeft > 0 && !g_squashActive) {
		if (g_impactLogFrame == 0) {
			printf("[Impact] logging %d frames after launch (every 3) | particle restitution %.2f%s | friction %.2f | object bounce %s | object shape %.3f | damping %.3f/frame | gravity %.3g | dt %.4f\n",
				g_impactLogLeft, g_objBodyContact ? 0.0f : g_groundRestitution, g_objBodyContact ? " (object bounce replaces it)" : "",
				g_groundFriction, g_objBodyContact ? "on" : "off", g_objShapeStiffness, velDamping, g_groundGravity, dt);
		}
		if (g_impactLogFrame % 3 == 0) logImpactMetrics(N, g_impactLogFrame, dt);
		++g_impactLogFrame;
		--g_impactLogLeft;
	}

	// [TN] 접선-법선 분해가 켜져 있으면 per-Gaussian J 를 매 프레임 갱신한다.
	//  (J Stats 토글과 독립 — 렌더가 이 값을 실제로 소비하므로 항상 최신이어야 한다)
	if (g_tnModeHost && d_volJScratch && d_volJPerG && d_detSigmaRest && d_matType &&
		d_volOffset && d_volRevOffset && d_volRevCount && d_volRevIdx) {
		computeVolumeJKernel << <blocks, threads >> > (
			N, d_pos_curr, d_volOffset, d_volCount, d_volIdx,
			d_detSigmaRest, d_matType, d_volJScratch);
		aggregateJPerGaussianKernel << <blocks, threads >> > (
			N, d_volJScratch, d_matType,
			d_volRevOffset, d_volRevCount, d_volRevIdx, d_volJPerG);
	}

	// J 통계 (수치 검증): 현재 pos_curr에 대해 J를 잰다. 제약 ON/OFF 어느 쪽이든.
	if (g_volCollectJStats && d_volJScratch && d_detSigmaRest && d_matType && d_volOffset) {
		computeVolumeJKernel << <blocks, threads >> > (
			N, d_pos_curr, d_volOffset, d_volCount, d_volIdx,
			d_detSigmaRest, d_matType, d_volJScratch);
		// CPU 감쇠 (100K는 400KB, 매 프레임 D2H로 부담 없음)
		std::vector<float> h_J(N);
		std::vector<int> h_mat(N);
		cudaMemcpy(h_J.data(), d_volJScratch, sizeof(float) * N, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_mat.data(), d_matType, sizeof(int) * N, cudaMemcpyDeviceToHost);
		std::vector<float> vals; vals.reserve(N);
		double sum = 0.0, sumSq = 0.0;
		float mn = FLT_MAX, mx = -FLT_MAX;
		// 부피 가중 누적 (Σ V·J, Σ V·J², Σ V). 개수 평균과 나란히 낸다.
		const bool haveVr = ((int)g_hVRestCache.size() == N);
		double wSum = 0.0, wJ = 0.0, wJ2 = 0.0;
		for (int i = 0; i < N; ++i) {
			if (h_mat[i] != 0) continue;
			const float j = h_J[i];
			if (!isfinite(j)) continue;
			vals.push_back(j);
			sum += j; sumSq += (double)j * j;
			mn = fminf(mn, j); mx = fmaxf(mx, j);
			if (haveVr) {
				const double v = (double)g_hVRestCache[i];
				if (v > 0.0 && isfinite(v)) { wSum += v; wJ += v * j; wJ2 += v * (double)j * j; }
			}
		}
		if (wSum > 0.0) {
			const double m = wJ / wSum;
			g_volJVwMean = (float)m;
			g_volJVwStd = (float)sqrt(fmax(0.0, wJ2 / wSum - m * m));
		}
		if (!vals.empty()) {
			const double n = (double)vals.size();
			const double mean = sum / n;
			const double var = fmax(0.0, sumSq / n - mean * mean);
			g_volJStatN = (int)vals.size();
			g_volJMean = (float)mean;
			g_volJStd = (float)sqrt(var);
			g_volJMin = mn;
			g_volJMax = mx;
			// 5/95 백분위 (tails 확인용) — nth_element로 O(n)
			const size_t p05 = (size_t)(0.05 * (n - 1));
			const size_t p95 = (size_t)(0.95 * (n - 1));
			std::nth_element(vals.begin(), vals.begin() + p05, vals.end());
			g_volJP05 = vals[p05];
			std::nth_element(vals.begin() + p05, vals.begin() + p95, vals.end());
			g_volJP95 = vals[p95];
		}

		// UI 버튼으로 예약된 1회 측정
		if (g_jSmoothPending) {
			measurePerGaussianJSmoothness(h_J, h_mat, N, g_jSmoothTag);
			g_jSmoothPending = false;
		}

		// 고정 변위 체크포인트 도달 + 정착 완료 → 자동 기록.
		// J 통계가 방금 갱신된 직후여야 하므로 반드시 이 자리에서 찍는다.
		if (g_squashLogPending) {
			char tag[96];
			snprintf(tag, sizeof(tag), "%s %3.0f%% disp=%.4f", g_squashPress ? "press" : "squash",
				(g_squashMaxDisp > 0.0f) ? (100.0f * g_squashCurDisp / g_squashMaxDisp) : 0.0f,
				g_squashCurDisp);
			FORWARD::logVolumeJStats(tag);
			logSquashShapeMetrics(N, tag);   // 옆으로 부푼 정도 + 따로 튀어나간 입자 (J 가 못 보는 것)
			// [진단] 접선-법선 분해용: per-Gaussian J 집계가 공간적으로 매끄러운지 같이 기록.
			// 하중(변위)이 고정된 이 시점이어야 조건 간 비교가 유효하다.
			measurePerGaussianJSmoothness(h_J, h_mat, N, tag);
			dumpFEMBenchmarkCheckpoint();
			g_squashLogPending = false;
		}
	}
	else if (g_squashLogPending) {
		logSquashShapeMetrics(N, g_squashPress ? "press (J stats off)" : "squash (J stats off)");
		// J 통계 수집이 꺼져 있으면 기록해봐야 이전 프레임 값이다. 조용히 버리고 경고.
		dumpFEMBenchmarkCheckpoint();
		g_squashLogPending = false;
		printf("[Squash] 경고: J Stats가 꺼져 있어 체크포인트를 기록하지 못했다.\n");
	}

	// 이 프레임의 클램프 발동 횟수를 읽어 누적
	if (g_volClampStats && g_useVolumeConstraint) {
		unsigned int c3[3] = { 0u, 0u, 0u };
		cudaMemcpyFromSymbol(c3, g_clampCtr, sizeof(c3));
		g_volClampAccum[0] += c3[0];
		g_volClampAccum[1] += c3[1];
		g_volClampAccum[2] += c3[2];
		++g_volClampFrames;
	}

	if (downloadToCPU) {
		downloadChainmailPositions(cm);
	}
}

static void runChainmailRelaxGPU(FORWARD::ChainMail& cm, int iterations, float stiffness, float damping, bool uploadFromCPU)
{
	ensureChainmailGPU(cm);
	if (uploadFromCPU) {
		uploadChainmailPositions(cm);
	}

	const int N = static_cast<int>(cm.numElements());
	const int threads = 256;
	const int blocks = (N + threads - 1) / threads;
	const bool useInertia = (g_cmInertiaGain > 0.0f);

	if (useInertia) {
		cudaMemcpy(d_pos_xpbd_tmp, d_pos_curr, sizeof(float3) * N, cudaMemcpyDeviceToDevice);
	}

	for (int it = 0; it < iterations; ++it) {
		chainmailRelaxKernel << <blocks, threads >> > (
			N,
			d_pos_curr,
			d_pos_next,
			d_density,
			d_offset,
			d_nbrCount,
			d_nbrIdx,
			d_nbrDist,
			d_nbrStiff,
			stiffness,
			damping,
			g_cmConstraintGlobalScale,
			g_cmConstraintAirScale,
			g_cmConstraintSkinScale,
			g_cmConstraintBoneScale,
			g_cmUseEdgeStiffness,
			g_cmEdgeStiffnessInfluence
			);
		std::swap(d_pos_curr, d_pos_next);
	}

	if (useInertia) {
		chainmailInertiaKernel << <blocks, threads >> > (
			N,
			d_pos_xpbd_tmp,
			d_pos_curr,
			d_vel,
			d_invMass,
			g_cmInertiaGain,
			g_cmVelocityRetention,
			g_cmVelocityClamp
			);
	}

	downloadChainmailPositions(cm);
}

static void runChainmailGPU(FORWARD::ChainMail& cm, int propIters, int relaxIters, float propStrength, float relaxStiffness, float relaxDamping, bool uploadFromCPU, bool downloadToCPU, const std::vector<int>* uploadIndices)
{
	//ensureChainmailGPU(cm);
	if (uploadFromCPU) {
		uploadChainmailPositions(cm);
	}
	else if (uploadIndices && !uploadIndices->empty()) {
		//uploadChainmailPositionsIndexed(cm, *uploadIndices);
	}

	const int N = static_cast<int>(cm.numElements());// 처리할 가우시안 개수
	const int threads = 256;// 블록당 스레드 수
	const int blocks = (N + threads - 1) / threads;// N 개를 256개 씩 나눠서 처리, 단 나머지가 있을 시 블록 하나 더 생성. 
	const bool useInertia = (g_cmInertiaGain > 0.0f);
	if (useInertia) {
		cudaMemcpy(d_pos_xpbd_tmp, d_pos_curr, sizeof(float3) * N, cudaMemcpyDeviceToDevice);
	}

	const bool useActiveMap = g_useActiveMap && !uploadFromCPU;
	if (useActiveMap) {
		if (g_activeMapReset) {
			cudaMemset(d_active_map, 0, sizeof(int) * N);
			cudaMemset(d_next_map, 0, sizeof(int) * N);
			g_activeMapReset = false;
		}

		for (int it = 0; it < propIters; ++it) {
			cudaMemcpy(d_time_next, d_time_curr, sizeof(float) * N, cudaMemcpyDeviceToDevice);
			cudaMemset(d_best_from, 0xFF, sizeof(int) * N);
			cudaMemset(d_next_map, 0, sizeof(int) * N);
			chainmailPropagateScatterKernel << <blocks, threads >> > (
				N,
				d_time_curr,
				d_time_next,
				d_density,
				d_offset,
				d_nbrCount,
				d_nbrIdx,
				d_active_map,
				d_best_from,
				d_next_map
				);
			applyBestFromKernel << <blocks, threads >> > (
				N,
				d_pos_curr,
				d_pos_next,
				d_density,
				d_offset,
				d_nbrCount,
				d_nbrIdx,
				d_nbrDist,
				d_nbrStiff,
				g_cmConstraintGlobalScale,
				g_cmConstraintAirScale,
				g_cmConstraintSkinScale,
				g_cmConstraintBoneScale,
				g_cmUseEdgeStiffness,
				g_cmEdgeStiffnessInfluence,
				d_best_from,
				d_next_map
				);
			std::swap(d_pos_curr, d_pos_next);
			std::swap(d_time_curr, d_time_next);
			std::swap(d_active_map, d_next_map);
		}

		for (int it = 0; it < relaxIters; ++it) {
			cudaMemset(d_next_map, 0, sizeof(int) * N);
			chainmailRelaxKernelActive << <blocks, threads >> > (
				N,
				d_pos_curr,
				d_pos_next,
				d_density,
				d_offset,
				d_nbrCount,
				d_nbrIdx,
				d_nbrDist,
				d_nbrStiff,
				relaxStiffness,
				relaxDamping,
				g_cmConstraintGlobalScale,
				g_cmConstraintAirScale,
				g_cmConstraintSkinScale,
				g_cmConstraintBoneScale,
				g_cmUseEdgeStiffness,
				g_cmEdgeStiffnessInfluence,
				d_active_map,
				d_next_map
				);
			std::swap(d_pos_curr, d_pos_next);
			std::swap(d_active_map, d_next_map);
		}

		if (useInertia) {
			chainmailInertiaKernel << <blocks, threads >> > (
				N,
				d_pos_xpbd_tmp,
				d_pos_curr,
				d_vel,
				d_invMass,
				g_cmInertiaGain,
				g_cmVelocityRetention,
				g_cmVelocityClamp
				);
		}
		cudaMemset(d_active_count, 0, sizeof(int));
		countActiveKernel << <blocks, threads >> > (N, d_active_map, d_active_count);
		cudaMemcpy(&g_lastActiveCount, d_active_count, sizeof(int), cudaMemcpyDeviceToHost);
		g_lastActiveRatio = (N > 0) ? (float)g_lastActiveCount / float(N) : 0.0f;
		return;
	}

	for (int it = 0; it < propIters; ++it) {
		chainmailPropagateKernelBest << <blocks, threads >> > (
			N,
			d_pos_curr,
			d_pos_next,
			d_time_curr,
			d_time_next,
			d_density,
			d_offset,
			d_nbrCount,
			d_nbrIdx,
			d_nbrDist,
			d_nbrStiff,
			propStrength,
			g_cmConstraintGlobalScale,
			g_cmConstraintAirScale,
			g_cmConstraintSkinScale,
			g_cmConstraintBoneScale,
			g_cmUseEdgeStiffness,
			g_cmEdgeStiffnessInfluence
			);
		std::swap(d_pos_curr, d_pos_next);
		std::swap(d_time_curr, d_time_next);
	}

	for (int it = 0; it < relaxIters; ++it) {
		chainmailRelaxKernel << <blocks, threads >> > (
			N,
			d_pos_curr,
			d_pos_next,
			d_density,
			d_offset,
			d_nbrCount,
			d_nbrIdx,
			d_nbrDist,
			d_nbrStiff,
			relaxStiffness,
			relaxDamping,
			g_cmConstraintGlobalScale,
			g_cmConstraintAirScale,
			g_cmConstraintSkinScale,
			g_cmConstraintBoneScale,
			g_cmUseEdgeStiffness,
			g_cmEdgeStiffnessInfluence
			);
		std::swap(d_pos_curr, d_pos_next);
	}

	if (useInertia) {
		chainmailInertiaKernel << <blocks, threads >> > (// 관성에대해서 효과를 주기위한 커널.
			N,
			d_pos_xpbd_tmp,
			d_pos_curr,
			d_vel,
			d_invMass,
			g_cmInertiaGain,
			g_cmVelocityRetention,
			g_cmVelocityClamp
			);
	}
	chainmailRelaxKernel << <blocks, threads >> > (
		N,
		d_pos_curr,
		d_pos_next,
		d_density,
		d_offset,
		d_nbrCount,
		d_nbrIdx,
		d_nbrDist,
		d_nbrStiff,
		relaxStiffness,
		relaxDamping,
		g_cmConstraintGlobalScale,
		g_cmConstraintAirScale,
		g_cmConstraintSkinScale,
		g_cmConstraintBoneScale,
		g_cmUseEdgeStiffness,
		g_cmEdgeStiffnessInfluence
		);
	std::swap(d_pos_curr, d_pos_next);
	g_lastActiveCount = N;
	g_lastActiveRatio = (N > 0) ? 1.0f : 0.0f;

	//if (downloadToCPU) {
	//	downloadChainmailPositions(cm);
	//}
}

#include <chrono>
#include <vector>
#include <numeric>
#include <iostream>
#include <algorithm>
// ── 외부 호스트(Isaac Sim 등)용: 변형된 가우시안 모양 (축 크기·방향) ─────────────────────
// 렌더 경로(preprocessCUDA 의 3D LS 블록 + computeCov3D2)와 같은 규칙으로 변형 후 공분산 Σ' 를 만들고,
// 그것을 USD ParticleField 가 받는 (scale 3축, 쿼터니언 w,x,y,z) 로 바꾼다.
//   A = (QPᵀ + εI)(PPᵀ + εI)⁻¹ (robust: 양쪽 ε, trace 정규화 역행렬) → AᵀA 고유분해 → 특이값 [1e-3, 1.5] 클램프
//   → R, S_final → M = S₀·R₀·S_final·Rᵀ,  Σ' = MᵀM          (렌더러가 그리는 공분산과 같은 식)
//   이동 ≤ deformEps, 이웃 < 3, 역행렬 실패, 비유한 값이면 원래 모양 그대로 (렌더 경로의 '변형 없음' 분기와 같다).
// Σ' 는 trace 로 정규화해 O(1) 에서 double Jacobi 로 고유분해한다 (얇은 가우시안의 작은 고유값을 살리려고).

// 대칭 3×3 double 고유분해 (cyclic Jacobi, 상대 멈춤). a 는 덮어쓴다. V 의 열이 고유벡터.
__device__ static void symEig3Double(double a[3][3], double ev[3], double V[3][3])
{
	for (int r = 0; r < 3; ++r)
		for (int c = 0; c < 3; ++c)
			V[r][c] = (r == c) ? 1.0 : 0.0;
	const int P[3] = { 0, 0, 1 };
	const int Q[3] = { 1, 2, 2 };
	for (int sweep = 0; sweep < 32; ++sweep) {
		const double off = fabs(a[0][1]) + fabs(a[0][2]) + fabs(a[1][2]);
		const double diag = fabs(a[0][0]) + fabs(a[1][1]) + fabs(a[2][2]);
		if (off <= 1e-15 * diag + 1e-300) break;
		for (int k = 0; k < 3; ++k) {
			const int p = P[k], q = Q[k];
			const double apq = a[p][q];
			if (fabs(apq) <= 1e-300) continue;
			const double theta = (a[q][q] - a[p][p]) / (2.0 * apq);
			const double t = ((theta >= 0.0) ? 1.0 : -1.0) / (fabs(theta) + sqrt(theta * theta + 1.0));
			const double c = 1.0 / sqrt(t * t + 1.0);
			const double s = t * c;
			for (int m = 0; m < 3; ++m) {   // A ← A·J (열)
				const double amp = a[m][p], amq = a[m][q];
				a[m][p] = c * amp - s * amq;
				a[m][q] = s * amp + c * amq;
			}
			for (int m = 0; m < 3; ++m) {   // A ← Jᵀ·A (행)
				const double apm = a[p][m], aqm = a[q][m];
				a[p][m] = c * apm - s * aqm;
				a[q][m] = s * apm + c * aqm;
			}
			for (int m = 0; m < 3; ++m) {   // V ← V·J
				const double vmp = V[m][p], vmq = V[m][q];
				V[m][p] = c * vmp - s * vmq;
				V[m][q] = s * vmp + c * vmq;
			}
		}
	}
	ev[0] = a[0][0];
	ev[1] = a[1][1];
	ev[2] = a[2][2];
}

__global__ void deformedShapeKernel(
	int N,
	const float3* rest,
	const float3* cur,
	const int* offset,
	const int* count,
	const int* nbr,
	const glm::vec3* scales,     // activated
	const glm::vec4* rots,       // 정규화, (w, x, y, z) = PLY rot_0..3
	float deformEps,
	int robust,
	float* outScale,             // [N*3] activated
	float* outQuat,              // [N*4] (w, x, y, z)
	unsigned int* ctr)           // [0] 원래 모양 그대로, [1] 변형 반영
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	const glm::vec3 s0 = scales[i];
	const glm::vec4 q0 = rots[i];
	const float3 x = cur[i];
	const float3 x0 = rest[i];
	const float dxv = x.x - x0.x, dyv = x.y - x0.y, dzv = x.z - x0.z;
	const float diff = sqrtf(dxv * dxv + dyv * dyv + dzv * dzv);
	const int kn = count[i];

	bool deformed = false;
	glm::mat3 Sigma(0.0f);
	if (diff > deformEps && kn >= 3) {
		const int off = offset[i];
		glm::mat3 PPt(0.0f), QPt(0.0f);
		for (int k = 0; k < kn; ++k) {
			const int j = nbr[off + k];
			if (j < 0 || j >= N) continue;
			const glm::vec3 p(rest[j].x - x0.x, rest[j].y - x0.y, rest[j].z - x0.z);
			const glm::vec3 q(cur[j].x - x.x, cur[j].y - x.y, cur[j].z - x.z);
			PPt += glm::outerProduct(p, p);
			QPt += glm::outerProduct(q, p);
		}
		const float tracePPt = PPt[0][0] + PPt[1][1] + PPt[2][2];
		const float alpha = (tracePPt > 0.0f) ? (tracePPt * 1e-4f) : 1e-6f;
		PPt += glm::mat3(alpha);
		if (robust) QPt += glm::mat3(alpha);
		glm::mat3 invPPt;
		bool ok;
		if (robust) {
			const float sP = fmaxf((PPt[0][0] + PPt[1][1] + PPt[2][2]) * (1.0f / 3.0f), 1e-30f);
			const float invSP = 1.0f / sP;
			glm::mat3 invHat;
			ok = inverse3x3_safe(PPt * invSP, invHat, 1e-9f);
			if (ok) invPPt = invHat * invSP;
		}
		else {
			ok = inverse3x3_safe(PPt, invPPt, 1e-8f);
		}
		if (ok) {
			const glm::mat3 A = QPt * invPPt;
			glm::mat3 AtA = glm::transpose(A) * A;
			glm::mat3 V;
			glm::vec3 S_squared;
			eigenDecomposition_glm(AtA, S_squared, V);
			glm::vec3 S_vec = glm::sqrt(glm::max(glm::vec3(0.0f), S_squared));
			S_vec = glm::min(glm::max(S_vec, glm::vec3(1e-3f)), glm::vec3(1.5f));
			const float eps = 1.0e-7f;
			const glm::vec3 S_inv(
				(S_vec.x > eps) ? 1.0f / S_vec.x : 0.0f,
				(S_vec.y > eps) ? 1.0f / S_vec.y : 0.0f,
				(S_vec.z > eps) ? 1.0f / S_vec.z : 0.0f);
			glm::mat3 U = A * V * glm::mat3(glm::vec3(S_inv.x, 0, 0), glm::vec3(0, S_inv.y, 0), glm::vec3(0, 0, S_inv.z));
			glm::mat3 R = U * glm::transpose(V);
			if (glm::determinant(R) < 0.0f) {
				U[0] *= -1.0f;
				R = U * glm::transpose(V);
			}
			const glm::mat3 S_final = V * glm::mat3(glm::vec3(S_vec.x, 0, 0), glm::vec3(0, S_vec.y, 0), glm::vec3(0, 0, S_vec.z)) * glm::transpose(V);
			// computeCov3D2 와 같은 식 (mod = 1)
			glm::mat3 S(1.0f);
			S[0][0] = s0.x;
			S[1][1] = s0.y;
			S[2][2] = s0.z;
			const float r = q0.x, qx = q0.y, qy = q0.z, qz = q0.w;
			const glm::mat3 R0(
				1.f - 2.f * (qy * qy + qz * qz), 2.f * (qx * qy - r * qz), 2.f * (qx * qz + r * qy),
				2.f * (qx * qy + r * qz), 1.f - 2.f * (qx * qx + qz * qz), 2.f * (qy * qz - r * qx),
				2.f * (qx * qz - r * qy), 2.f * (qy * qz + r * qx), 1.f - 2.f * (qx * qx + qy * qy));
			const glm::mat3 M = S * R0 * S_final * glm::transpose(R);
			Sigma = glm::transpose(M) * M;
			deformed = true;
			for (int c = 0; c < 3; ++c)
				for (int rr = 0; rr < 3; ++rr)
					if (!isfinite(Sigma[c][rr])) deformed = false;
		}
	}

	double tr = deformed ? ((double)Sigma[0][0] + Sigma[1][1] + Sigma[2][2]) : 0.0;
	if (!deformed || !(tr > 0.0)) {
		outScale[3 * i + 0] = s0.x;
		outScale[3 * i + 1] = s0.y;
		outScale[3 * i + 2] = s0.z;
		outQuat[4 * i + 0] = q0.x;
		outQuat[4 * i + 1] = q0.y;
		outQuat[4 * i + 2] = q0.z;
		outQuat[4 * i + 3] = q0.w;
		atomicAdd(&ctr[0], 1u);
		return;
	}

	// Σ' = R diag(λ) Rᵀ  →  scale = √λ, 방향 = R (열 = 가우시안 로컬 축의 월드 방향)
	const double inv = 3.0 / tr;
	double a[3][3];
	for (int rr = 0; rr < 3; ++rr)
		for (int c = 0; c < 3; ++c)
			a[rr][c] = 0.5 * ((double)Sigma[c][rr] + (double)Sigma[rr][c]) * inv;
	double ev[3], E[3][3];
	symEig3Double(a, ev, E);
	const double det = E[0][0] * (E[1][1] * E[2][2] - E[1][2] * E[2][1])
		- E[0][1] * (E[1][0] * E[2][2] - E[1][2] * E[2][0])
		+ E[0][2] * (E[1][0] * E[2][1] - E[1][1] * E[2][0]);
	if (det < 0.0) {
		for (int rr = 0; rr < 3; ++rr) E[rr][2] = -E[rr][2];
	}
	// 하한: 원래 가장 긴 축의 1e-3 (렌더 경로의 특이값 하한과 같은 비율). 한 방향으로 거의 납작해진 가우시안은
	// 클램프된 S_inv 로 만든 R 이 찌그러져(the covariance-stability guard) Σ' 한 축이 0 이 된다 — 두께 0 타원체를 레이트레이서에 넘기지 않는다.
	const double sFloor = 1e-3 * (double)fmaxf(s0.x, fmaxf(s0.y, s0.z));
	for (int k = 0; k < 3; ++k) outScale[3 * i + k] = (float)fmax(sqrt(fmax(ev[k], 0.0) / inv), sFloor);

	// 회전행렬 E → 쿼터니언 (w, x, y, z)
	double w, qxv, qyv, qzv;
	const double t = E[0][0] + E[1][1] + E[2][2];
	if (t > 0.0) {
		const double s = 2.0 * sqrt(t + 1.0);
		w = 0.25 * s;
		qxv = (E[2][1] - E[1][2]) / s;
		qyv = (E[0][2] - E[2][0]) / s;
		qzv = (E[1][0] - E[0][1]) / s;
	}
	else if (E[0][0] > E[1][1] && E[0][0] > E[2][2]) {
		const double s = 2.0 * sqrt(1.0 + E[0][0] - E[1][1] - E[2][2]);
		w = (E[2][1] - E[1][2]) / s;
		qxv = 0.25 * s;
		qyv = (E[0][1] + E[1][0]) / s;
		qzv = (E[0][2] + E[2][0]) / s;
	}
	else if (E[1][1] > E[2][2]) {
		const double s = 2.0 * sqrt(1.0 + E[1][1] - E[0][0] - E[2][2]);
		w = (E[0][2] - E[2][0]) / s;
		qxv = (E[0][1] + E[1][0]) / s;
		qyv = 0.25 * s;
		qzv = (E[1][2] + E[2][1]) / s;
	}
	else {
		const double s = 2.0 * sqrt(1.0 + E[2][2] - E[0][0] - E[1][1]);
		w = (E[1][0] - E[0][1]) / s;
		qxv = (E[0][2] + E[2][0]) / s;
		qyv = (E[1][2] + E[2][1]) / s;
		qzv = 0.25 * s;
	}
	const double qn = sqrt(w * w + qxv * qxv + qyv * qyv + qzv * qzv);
	outQuat[4 * i + 0] = (float)(w / qn);
	outQuat[4 * i + 1] = (float)(qxv / qn);
	outQuat[4 * i + 2] = (float)(qyv / qn);
	outQuat[4 * i + 3] = (float)(qzv / qn);
	atomicAdd(&ctr[1], 1u);
}

// 현재 위치(d_pos_curr)와 rest 로 변형된 가우시안 모양을 계산해 디바이스 버퍼에 쓴다.
//  scales / rotations : 입력 (activated scale, 정규화 쿼터니언 w,x,y,z) 디바이스 포인터
//  outScales[N*3], outQuats[N*4] : 출력 디바이스 버퍼.  deformedCount 에 변형이 반영된 가우시안 수.
bool FORWARD::computeDeformedShapes(const glm::vec3* scales, const glm::vec4* rotations,
	float* outScales, float* outQuats, float deformEps, int* deformedCount)
{
	const int N = cm_num_elements;
	if (N <= 0 || !d_pos_rest || !d_pos_curr || !d_offset || !d_nbrCount || !d_nbrIdx ||
		!scales || !rotations || !outScales || !outQuats) return false;
	static unsigned int* d_ctr = nullptr;
	if (!d_ctr) cudaMalloc(&d_ctr, sizeof(unsigned int) * 2);
	cudaMemset(d_ctr, 0, sizeof(unsigned int) * 2);
	const int threads = 256;
	deformedShapeKernel << <(N + threads - 1) / threads, threads >> > (
		N, d_pos_rest, d_pos_curr, d_offset, d_nbrCount, d_nbrIdx, scales, rotations,
		deformEps, g_renderFRobustHost ? 1 : 0, outScales, outQuats, d_ctr);
	unsigned int c[2] = { 0u, 0u };
	cudaMemcpy(c, d_ctr, sizeof(c), cudaMemcpyDeviceToHost);
	if (deformedCount) *deformedCount = (int)c[1];
	return cudaGetLastError() == cudaSuccess;
}

// ── 외부 호스트(Isaac Sim 등)용 물리 진입점 ──────────────────────────────────────
// preprocess 의 물리 부분(ensureChainmailGPU → runXPBDSimulation)만 떼어낸 것. 렌더·마우스 피킹(applySeedCommands)·
// 렌더 F 는 하지 않는다. scales / rotations 는 부피 제약 mixture rest 용 디바이스 포인터로 preprocess 와 같은 규약
// (activated scale, 정규화 쿼터니언 PLY rot_0..3 순서). 뷰어는 이 함수를 쓰지 않는다.
void FORWARD::stepPhysicsOnly(FORWARD::ChainMail& cm, const glm::vec3* scales, const glm::vec4* rotations)
{
	g_gsScalesPtr = scales;
	g_gsRotationsPtr = rotations;
	ensureChainmailGPU(cm);
	const float3 gravity = (g_groundGravity > 0.0f && !g_squashActive)
		? make_float3(-g_groundGravity * g_groundN[0], -g_groundGravity * g_groundN[1], -g_groundGravity * g_groundN[2])
		: make_float3(0.0f, 0.0f, 0.0f);
	if (!g_groundPaused || g_squashActive) {
		runXPBDSimulation(cm, g_xpbdSolverIters, g_xpbdDt, g_xpbdUnderRelax, g_xpbdVelDamping, gravity, false, false, nullptr);
	}
}

// 현재 위치 디바이스 버퍼 (float3[N], 매 스텝 커밋된 x_{t+1}). 순서 = ChainMail 원소 순서.
const float* FORWARD::getPhysicsPositionsDevice(int* count)
{
	if (count) *count = cm_num_elements;
	return reinterpret_cast<const float*>(d_pos_curr);
}

// 다음 스텝에서 그래프와 버퍼를 다시 올리게 한다. ensureChainmailGPU 는 노드·이웃 수만 보고 건너뛰므로,
// 같은 크기의 다른 그래프를 새로 불러올 때 필요하다.
void FORWARD::invalidatePhysicsGraph() { chainmailInit = false; }

void FORWARD::preprocess(
	FORWARD::ChainMail& cm,
	std::vector<int>& activeSet,

	int P, int D, int M,
	 float* means3D,
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
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	int2* rects,
	float3 boxmin,
	float3 boxmax,
	bool antialiasing,float t,
	bool _wave,
	bool _twist,
	bool _bubble
	)
{
	const bool useGpuChainmail = (g_GpuChainmailMode == 1);
	const bool useXPBD = (g_physicsMode == 1);
	// mixture rest(명찰/신분증)용 렌더 속성 포인터를 저장해둔다.
	// 이 아래에서 호출되는 ensureChainmailGPU → rebuildVolumeClusters가 사용한다.
	g_gsScalesPtr = scales;
	g_gsRotationsPtr = rotations;
	//
	const int cmRelaxIters = g_cmRelaxIters;
	const float cmStiffness = g_cmStiffness;
	const float cmDamping = g_cmDamping;
	const int cmPropIters = g_cmPropIters;
	const float cmPropStrength = g_cmPropStrength;
	//
	const int xpbdSolverIters = g_xpbdSolverIters;
	const float xpbdDt = g_xpbdDt;//XPBD 
	const float xpbdUnderRelax = g_xpbdUnderRelax;
	const float xpbdVelDamping = g_xpbdVelDamping;//속도, 마찰 즉 점들이 끌려갈때 생기는 관성(속도) 얼마나빠르게할지 클수록 빨라짐
	// 중력은 Ground 데모가 켰을 때만 (−up 방향). 끄면 기존과 같은 0 벡터.
	const float3 xpbdGravity = (g_groundGravity > 0.0f && !g_squashActive)   // squash 실험에는 중력을 넣지 않는다
		? make_float3(-g_groundGravity * g_groundN[0], -g_groundGravity * g_groundN[1], -g_groundGravity * g_groundN[2])
		: make_float3(0.0f, 0.0f, 0.0f);
	//
	const bool cmSyncCPU = false;
	setGpuCommandMode(useGpuChainmail && !cmSyncCPU);
	// ==========================================
	// [측정용 변수] 함수 내부에 static으로 선언
	// ==========================================
	static std::vector<float> timeLog_Prop;
	static std::vector<float> timeLog_Cov;
	static std::vector<float> timeLog_Input;
	static int measureCount = 0;

	// 측정 시작: Propagation (ChainMail)
	static cudaEvent_t ev_prop_start = nullptr;
	static cudaEvent_t ev_prop_end = nullptr;
	static cudaEvent_t ev_cov_start = nullptr;
	static cudaEvent_t ev_cov_end = nullptr;
	static bool ev_init = false;
	if (!ev_init) {
		cudaEventCreate(&ev_prop_start);
		cudaEventCreate(&ev_prop_end);
		cudaEventCreate(&ev_cov_start);
		cudaEventCreate(&ev_cov_end);
		ev_init = true;
	}
	auto cpu_input_start = std::chrono::high_resolution_clock::now();
	float cpu_input_ms = 0.0f;
	float cpu_prop_ms = 0.0f;
	bool prop_timing_valid = false;
	// ----------------------------------------------------------------
	// (1) ChainMail Propagation 로직 (기존 코드)
	// ----------------------------------------------------------------
	std::vector<int> frameActive;

	static float prevD = 0.0f;
	static bool applyForce = true;
	float amp = 0.004f;
	float speed = 8.0f;
	float d = amp * sin(t * speed);
	float maxAngle = 0.02f;
	float angle = maxAngle * sin(t * speed); // 현재 프레임의 회전 각도
	if (cm.singleDeformTask.isRunning) {
	cm.startWave(cm.singleDeformTask.seedIdx, glm::vec3(d, 0, 0), activeSet);
	}

	else {
		// ... (기존 else 구문 유지) ...
		if (useGpuChainmail) {
			if (!cmSyncCPU) {
				cpu_input_start = std::chrono::high_resolution_clock::now();
				ensureChainmailGPU(cm);
				//if (useXPBD) {
					applySeedCommands(cm);
				//}
				//else {
				//	applySeedCommandsGPU(cm);
				//}
				cpu_input_ms = std::chrono::duration<float, std::milli>(
					std::chrono::high_resolution_clock::now() - cpu_input_start).count();
			}
			cudaEventRecord(ev_prop_start);
			if (useXPBD) {
				if (!g_groundPaused || g_squashActive) runXPBDSimulation(cm, xpbdSolverIters, xpbdDt, xpbdUnderRelax, xpbdVelDamping, xpbdGravity, cmSyncCPU, cmSyncCPU, nullptr); // Ground 데모 일시정지: 물리 스텝만 건너뛰고 렌더는 계속
			}
			else {
				runChainmailGPU(cm, cmPropIters, cmRelaxIters, cmPropStrength, cmStiffness, cmDamping, cmSyncCPU, cmSyncCPU, nullptr);
			}
			cudaEventRecord(ev_prop_end);
			cudaEventSynchronize(ev_prop_end);
			prop_timing_valid = true;
		}
		else {
			//cpu cm
			auto cpu_prop_start = std::chrono::high_resolution_clock::now();
			cm.propagate(frameActive);
			activeSet.insert(activeSet.end(), frameActive.begin(), frameActive.end());
			std::sort(activeSet.begin(), activeSet.end());
			activeSet.erase(std::unique(activeSet.begin(), activeSet.end()), activeSet.end());
			cm.relax(activeSet);
			cpu_prop_ms = std::chrono::duration<float, std::milli>(
				std::chrono::high_resolution_clock::now() - cpu_prop_start).count();
		}
	}
	
	// 측정 종료: Propagation
	

	// 측정 시작: Covariance Update (CUDA Kernel)
	cudaEventRecord(ev_cov_start);
	//activeSet.clear();  // 프레임 단위 reset

	//if (cm.isWaveRunning())
		//cm.propagateStep(activeSet);

	//cm.propagate(activeSet);
	//cm.relax(activeSet);

	// 렌더가 매 프레임 읽는 변형 후 위치/시간 버퍼.
	// P가 바뀌면(모델 재로드 등) 다시 잡는다. 예전에는 한 번만 할당해 stale 상태가 되었다.
	static float* d_means3D = nullptr;
	static float* nbr_time = nullptr;
	static int cached_P = 0;
	if (cached_P != P) {
		if (d_means3D) cudaFree(d_means3D);
		if (nbr_time) cudaFree(nbr_time);
		cudaMallocManaged(&d_means3D, sizeof(float) * 3 * P);
		cudaMallocManaged(&nbr_time, sizeof(float) * P);
		cached_P = P;
	}

	// 변형 그래디언트 커널이 물리 커널과 같은 CSR을 읽으므로, CPU 모드에서도 토폴로지가
	// 디바이스에 올라와 있어야 한다. ensureChainmailGPU는 이미 올라와 있으면 즉시 반환한다.
	const bool graphMatchesP = ((int)cm.numElements() == P);
	if (graphMatchesP) {
		ensureChainmailGPU(cm);
	}
	const bool hasGraph = graphMatchesP && d_offset && d_nbrCount && d_nbrIdx;

	// Expose the latest deformed positions buffer for debug visualization.
	g_latest_deformed_xyz = d_means3D;
	g_latest_deformed_count = P;
	// Expose the exact render-path 2D projection buffers for debug visualization.
	g_latest_means2d = means2D;
	g_latest_radii = radii;
	g_latest_project_count = P;
	g_latest_render_w = W;
	g_latest_render_h = H;
	if (hasGraph && useGpuChainmail) {
		const int threads = 256;
		const int blocks = (P + threads - 1) / threads;
		PackKernel << <blocks, threads >> > (
			P,
			d_pos_curr,
			d_time_curr,
			d_means3D,
			nbr_time
			);
		cudaDeviceSynchronize();
	}
	else if (hasGraph) {
		for (int i = 0; i < P; ++i) {
			const Element& E = cm.getElement(i);
			d_means3D[3 * i + 0] = E.pos.x;
			d_means3D[3 * i + 1] = E.pos.y;
			d_means3D[3 * i + 2] = E.pos.z;
			nbr_time[i] = E.time;
		}
		cudaDeviceSynchronize();
	}
	else {
		// 그래프가 아직 없거나 가우시안 수와 어긋난 경우: 변형 없이 원본 위치로 렌더한다.
		cudaMemcpy(d_means3D, means3D, sizeof(float) * 3 * P, cudaMemcpyDeviceToDevice);
		cudaMemset(nbr_time, 0, sizeof(float) * P);
		cudaDeviceSynchronize();
	}

	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		d_means3D,
		means3D,
		// --- neighborhood for 3D F (CSR, 물리 커널과 동일 버퍼) ---
		hasGraph ? d_offset : nullptr,
		hasGraph ? d_nbrCount : nullptr,
		hasGraph ? d_nbrIdx : nullptr,
		nbr_time,
		scales,
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
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered,
		rects,
		boxmin,
		boxmax,
		antialiasing,
		t,
		_wave,
		_twist,
		_bubble,
		// J 시각화: XPBD 모드에서만 J 버퍼가 매 프레임 갱신되므로 그때만 넘긴다
		(g_volJVizEnabled && useXPBD && g_volPrecomputed && d_volJScratch) ? d_volJScratch : nullptr,
		g_volJVizGain,
		// matType 시각화: 그래프 로드 후 언제든 (물리 모드 무관)
		(g_volMatVizEnabled && g_volPrecomputed && d_matType) ? d_matType : nullptr
		);
	cudaDeviceSynchronize();

	// 측정 종료: Covariance Update
	cudaEventRecord(ev_cov_end);
	cudaEventSynchronize(ev_cov_end);

	if(cm.FPS){
	
		// ==========================================
		// [로그 출력 로직]
		// ==========================================
		// 상호작용이 있을 때만(isWaveRunning) 혹은 항상 측정할지 결정
		float ms_prop_gpu = 0.0f;
		float ms_cov_gpu = 0.0f;
		if (useGpuChainmail && prop_timing_valid) {
			cudaEventElapsedTime(&ms_prop_gpu, ev_prop_start, ev_prop_end);
		}
		cudaEventElapsedTime(&ms_cov_gpu, ev_cov_start, ev_cov_end);
		timeLog_Prop.push_back(useGpuChainmail ? ms_prop_gpu : cpu_prop_ms);
		timeLog_Cov.push_back(ms_cov_gpu);
		timeLog_Input.push_back(cpu_input_ms);
		measureCount++;

		// 100 프레임마다 평균 출력 (너무 자주 출력하면 느려짐)
		if (cm.FPS && measureCount >= 100) {
			float sum_prop = std::accumulate(timeLog_Prop.begin(), timeLog_Prop.end(), 0.0f);
			float sum_cov = std::accumulate(timeLog_Cov.begin(), timeLog_Cov.end(), 0.0f);
			float sum_input = std::accumulate(timeLog_Input.begin(), timeLog_Input.end(), 0.0f);
			float avg_prop = sum_prop / timeLog_Prop.size();
			float avg_cov = sum_cov / timeLog_Cov.size();
			float avg_input = sum_input / timeLog_Input.size();
			const char* propLabel = useGpuChainmail ? (useXPBD ? "XPBD (GPU)" : "ChainMail (GPU)") : "ChainMail (CPU)";

			printf("\n==============================================\n");
			printf(" [ Performance Analysis - Avg of 100 Frames ] \n");
			printf(" * Physics (%s) : %.4f ms\n", propLabel, avg_prop);//ChainMail 전파 및 XPBD 솔버 연산 시간. 물리적 복잡도
			printf(" * Covariance (GPU) : %.4f ms\n", avg_cov);//3DGS 기하학적 업데이트 시간
			if(!useGpuChainmail)printf(" * Input (CPU)       : %.4f ms\n", avg_input);//CPU에서 GPU로 데이터를 준비/전송하는 시간
			const float total_ms = avg_prop + avg_cov + avg_input;
			printf(" * Total Latency         : %.4f ms\n", total_ms);//위 세 가지를 합친 전체 파이프라인 시간. 한 프레임의 연산이 완료되는 총 시간
			printf(" * Est. FPS              : %.2f FPS", 1000.0f / total_ms);
			printf("\n==============================================\n");
			// 클램프 계측: 프레임당 평균 발동 횟수. 0에 가까울수록 솔버가 안정적(클램프 불필요).
			if (g_volClampStats && g_volClampFrames > 0) {
				const double f = (double)g_volClampFrames;
				printf(" [Clamp/frame] J=%.1f  step=%.1f  eig(G4)=%.1f  (부피 volume=%d개 중)\n",
					g_volClampAccum[0] / f, g_volClampAccum[1] / f, g_volClampAccum[2] / f,
					g_volMatCount[0]);
				g_volClampAccum[0] = g_volClampAccum[1] = g_volClampAccum[2] = 0;
				g_volClampFrames = 0;
			}
			printf("\n");

			timeLog_Prop.clear();
			timeLog_Cov.clear();
			timeLog_Input.clear();
			measureCount = 0;
		}



	}else {
		// 버튼 껐을 때는 데이터 초기화 (찌꺼기 데이터 방지)
		if (!timeLog_Prop.empty()) {
			timeLog_Prop.clear();
			timeLog_Cov.clear();
			timeLog_Input.clear();
			measureCount = 0;
		}
	}
	
	

}
