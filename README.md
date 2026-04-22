# AI Semiconductor Cycle Intelligence

> **무료 API만으로 AI/반도체 강세장 사이클 위치를 판단하는 서버리스 시스템**

CCI(crypto-cycle-intelligence)와 동일한 아키텍처를 주식 시장에 적용한 자매 시스템입니다.

## 🎯 ASCS (AI Semiconductor Cycle Score)

22개 지표를 6차원으로 가중 합산하여 0-100점으로 사이클 위치를 표현합니다:

| Phase | Range | 의미 |
|---|---|---|
| 🧊 Capitulation | 0-20 | 극단적 매도, 패닉. 역사적 매수 기회 |
| 🌱 Recovery     | 20-40 | 바닥 형성, 펀더멘털 개선 |
| 📈 Expansion    | 40-60 | 정상 상승 사이클 |
| 🔥 Late Bull    | 60-80 | 과열 신호, 집중 위험 |
| 🚨 Euphoria     | 80-100 | 거품 단계 |

## 📊 6차원 22지표

| 차원 | 가중치 | 주요 지표 |
|---|---|---|
| Valuation | 25% | NVDA Forward P/E, P/S, SOX 평균 P/E 백분위, Top10 칩 시총/GDP |
| Earnings  | 20% | NVDA 매출 성장률, 반도체 평균 매출 성장, Hyperscaler capex YoY |
| Capital   | 15% | Memory capex YoY, Semi capex/매출 비율 |
| Sentiment | 15% | VIX, SOX 200일선 위 %, NVDA short% |
| Macro     | 10% | 10Y 수익률, DXY, ISM PMI, M2 YoY |
| Technical | 15% | SOX Weekly RSI, Daily RSI, NVDA 52주 위치, SOX 200D 거리 |

## 🏗️ 아키텍처

```
GitHub Actions (cron) → JSON commit → GitHub Pages (대시보드) + Telegram
                                              ↑
                                     Yahoo Finance + FRED
```

- **TimescaleDB 없음**: Git이 DB
- **백엔드 없음**: 정적 JSON
- **로컬 설치 없음**: GitHub 안에서 모든 게 동작
- **월 비용 $0**

## 📂 파일 구조

```
asci-serverless/
├── scripts/run_pipeline.py    파이프라인 (yfinance + FRED → ASCS → JSON + Telegram)
├── docs/site/index.html       대시보드 (단일 HTML, React CDN)
├── .github/workflows/pipeline.yml  Actions cron
├── data/                       자동 갱신 JSON
│   ├── latest.json             최신 스냅샷
│   ├── history.json            히스토리
│   └── snapshots/YYYY-MM-DD.json
└── requirements.txt
```

## 🌐 데이터 소스

- **yfinance** — 가격, 시총, 재무, 기술적 지표 (무료, 키 불필요)
- **FRED API** — 매크로 (10Y, DXY, M2, ISM PMI), 무료 키만 필요

## 🚀 운영 스케줄

- **장 시작** (09:30 ET / 14:30 UTC): 가격 갱신
- **장 중간** (12:30 ET / 17:30 UTC): 갱신
- **장 마감** (16:00 ET / 21:05 UTC): 마감 데이터
- **매일 07:00 KST** (22:00 UTC): 메인 일일 리포트 (텔레그램)

## 📈 추적 종목

**Semiconductor Leaders** (10개): NVDA, AVGO, TSM, AMD, MU, ASML, AMAT, KLAC, LRCX, INTC
**Hyperscalers** (5개): MSFT, AMZN, GOOGL, META, ORCL
**Indices** (7개): ^SOX, SOXX, SMH, ^GSPC, ^NDX, ^VIX, DXY

---

*v1.0 · Serverless · 검증된 CCI 패턴 적용*
