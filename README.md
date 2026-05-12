# 🏗️ RIA — Risk Instinct Alert
> 기후 누적 스트레스 기반 건설안전사고 예보 시스템  
> 제5회 AI·공공데이터 활용 및 창업 경진대회 공모전 출품작

---

## 📌 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 프로젝트명 | Risk Instinct Alert (RIA) |
| 기관 선정 | 국토안전관리원 |
| 분석 기간 | 2019년 7월 ~ 2025년 12월 |
| 최종 데이터셋 | RIA_최종통합피쳐셋.csv (40,714행 × 53컬럼) |

---

## 🎯 핵심 아이디어

단발성 당일 기상 정보가 아닌 **7일 누적 기후 스트레스**를 정량화하여 건설현장 위험을 사전 예보합니다.

- 콘크리트 양생 기간(7~8일): 온도 변화에 민감
- 구조물 열팽창 누적 주기: 일주일 단위 온도 편차 누적이 구조물에 영향
- 근로자 열피로 축적 패턴: 3~7일 지속 노출 후 위험도 급증

---

## 🏗️ 시스템 아키텍처

```
[오프라인 학습]
사고 이력 데이터 (2019-2024)
    ↓  평년 편차 피처 계산
    ↓  14일 기상 윈도우 구성
    ↓  카테고리 인코딩 + 정규화
    ↓  Negative Sampling (3:1)
    ↓  Encoder-Decoder Transformer 학습 (8 epochs)
    ↓  검증 데이터로 캘리브레이션
→  ria_model.pt 저장

[온라인 추론]
KMA API → 오늘 날씨
    ↓  누적 피처 계산 (7d 롤링)
    ↓  시나리오 분류
    ↓  환경 타입 × 모델 추론 (6번)
    ↓  Lift 필터링 + 카테고리 매핑
    ↓  날씨 룰 이슈 병합
→  체크리스트 → Slack
```

---

## 📂 Repository 구조

```
RIA/
├── README.md
├── docs/
│   ├── RIA_결과보고서_제출용.docx     # 최종 제출 보고서
│   └── RIA_webapp.html               # 웹 데모 (브라우저에서 바로 실행)
├── data/
│   └── (RIA_최종통합피쳐셋.csv)       # 용량 이슈로 별도 공유
├── notebooks/
│   └── EDA_모델링.ipynb              # EDA 및 모델링 탐색
└── src/
    └── (팀원 소스코드 참고)
        ├── 13_train_model.py
        ├── 14_inference_demo.py
        ├── 16_run_alert.py
        └── ria_review_server.py
```

---

## 📊 데이터 파이프라인

### 원본 데이터 출처

| 데이터 | 출처 | 규모 |
|---|---|---|
| 건설안전사고 발생현황 (재해자특성) | 국토안전관리원 빅토리 플랫폼 | 40,714건 × 30컬럼 |
| 건설안전사고 발생현황 (소분류별) | 국토안전관리원 빅토리 플랫폼 | 65건 × 12컬럼 |
| 일별 전국 기상 데이터 | 기상청 | 2019.01~2025.12 |

### 피처 엔지니어링

**기상 파생변수 (11개)**

| 변수명 | 산출 방식 | 설계 근거 |
|---|---|---|
| ATD_7d | 최근 7일간 (평균기온 - 평년기온) 합계 | 구조물 열팽창 누적 주기 |
| TSI_3d | 최근 3일 일교차 × 시간가중치(0.5/0.33/0.17) | 단기 온도 급변 충격 정량화 |
| 누적강수_7d | 최근 7일 일강수량 합계 | 지반 연약화 누적 지표 |
| 기온변동성_7d | 최근 7일 평균기온 표준편차 | 온도 급변 불안정성 |
| 폭염/한파/강수/강풍_연속_7d | 7일 내 임계조건 초과 일수 | 극한 기상 지속성 |
| magnitude_score | 위 변수들 가중합산 후 Min-Max 정규화 | 복합 기상 위험 종합 지표 |

**교차 파생변수 (7개)**

| 변수명 | 설계 근거 |
|---|---|
| risk_heat_process | ATD_7d × 고위험공종여부 |
| risk_rain_progress | 누적강수_7d × (1 - 공정율/100) |
| risk_cold_concrete | 한파_연속_7d × 콘크리트공사 여부 |
| risk_wind_scaffold | 누적풍속_7d × 고위험공종여부 |
| weather_scale_interact | magnitude_score × (1/공사비_순위) |
| heat_foreign_exposure | 폭염_연속_7d × 외국인비율 |
| heat_elderly_exposure | 폭염_연속_7d × 고령자비율 |

---

## 🤖 모델

### Encoder-Decoder Transformer

```
Encoder (날씨 패턴 학습)
- 입력: (Batch, 14일, 10개 피처)
- nn.Linear(10 → 64)
- Positional Embedding
- TransformerEncoder (2 layers, 4 heads)

Decoder (현장 상황 × 날씨 패턴 종합 판단)
- 현장 컨텍스트 → Query 벡터
- Cross-Attention with Encoder output (Key, Value)
- 출력: 이슈별 위험 확률 [0, 1]
```

### 학습 설정

| 항목 | 설정값 |
|---|---|
| Optimizer | AdamW (weight_decay=1e-4) |
| Learning rate | 1e-3 |
| Scheduler | CosineAnnealingLR(T_max=8) |
| Epochs | 8 |
| Batch size | 512 |
| 불균형 처리 | pos_weight=23 (BCEWithLogitsLoss) |
| 검증 분할 | 2025-01-01 기준 시계열 분할 |

### Label Leakage 처리

학습 시 아래 컬럼 전량 제외:
- `magnitude_score`, `risk_*`, `weather_scale_interact`, `heat_*_exposure`
- 사고유형 7종 플래그 (Layer 1 레이블)
- 사상자 컬럼 전체 (재해자 특성)
- `C_RPI`, `위험등급`

---

## 📈 주요 분석 인사이트

### 날씨 × 공종 복합 신호

| 조건 | 공종 | 중대재해율 | Lift |
|---|---|---|---|
| 강수_연속 ≥ 3일 | 철골공사 | 10.98% | 2.67x |
| 폭염_연속 ≥ 2일 | 토공사 | 9.28% | 2.26x |
| 최저기온 ≤ -5°C | 철골공사 | 7.69% | 1.87x |
| 누적강수_7d ≥ 100mm | 기계설비공사 | 7.55% | 1.83x |

> 날씨 단독보다 공종과 결합 시 Lift가 크게 상승 →  
> **날씨가 위험을 직접 만드는 것이 아니라 특정 공종의 취약점을 증폭시킴**

### 소규모 현장 집중 위험

- 공사비 50억 미만 소규모 현장 중대재해율: **10.1%** (대형 현장 2.9%의 3.5배)
- 전체 중대재해 1,675건 중 778건(46.4%)이 소규모 현장 발생

---

## 🚨 C-RPI (건설현장 누적 위험지수)

| 등급 | 범위 | 현장 조치 |
|---|---|---|
| 🟢 관심 | 0 ~ 25점 | 일반 안전 관리 유지 |
| 🟡 주의 | 26 ~ 50점 | 작업 전 기상 상황 재확인 |
| 🟠 경계 | 51 ~ 75점 | 고위험 공종 작업 중단 검토 |
| 🔴 심각 | 76 ~ 100점 | 현장 전면 작업 중단 |

---

## 💬 Slack 알림 연동

```python
# Bot Token 방식
payload = {
    "channel": "#시설안전-알림",
    "text": f"🚨 [RIA 위험 감지] C-RPI {score}점",
    "blocks": [...]  # Block Kit 형식
}
response = requests.post(
    "https://slack.com/api/chat.postMessage",
    headers={"Authorization": f"Bearer {SLACK_TOKEN}"},
    json=payload
)
```

---

## ⚠️ 데이터 구조적 한계

현 데이터셋의 핵심 한계는 **전국 단위 평균 기상값 사용**으로, 세 가지 문제가 파생됨.

- 지역별 실제 기상 조건 미반영 → 국지성 기상 이벤트 과소평가 가능
- 강풍_연속_7d 전체 0 고정 → 전국 평균 풍속이 임계값(9m/s) 미달
- 수도권 외 지역 표본 부족 → 지역 분리 분석 불가

**시도별 관측소 데이터 연계** 시 세 문제 동시 해소 가능.

---

## 🔗 References

- [국토안전관리원 빅토리 플랫폼](https://bigtori.kalis.or.kr)
- [고용노동부 2024 산업재해 현황](http://office.shinwooi.co.kr/board/download3.asp?type=2&category=7&seq=351)
- [중대재해처벌법 50억 미만 확대 (2024.1)](https://kiramonthly.com/1854)

---

## 👥 팀

제5회 AI·공공데이터 활용 및 창업 경진대회  
**AI·데이터를 활용한 서비스 발굴 부문 | 국토안전관리원**
