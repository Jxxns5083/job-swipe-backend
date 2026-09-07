import os
import json
import asyncio
import httpx
from bs4 import BeautifulSoup
from typing import List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from openai import OpenAI

# 1. 서버 및 클라이언트 초기화
app = FastAPI(title="Job Swipe App API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
ai_client = OpenAI(api_key=OPENAI_API_KEY)

class SwipeActionRequest(BaseModel):
    user_id: str
    job_id: str
    action: str

# 2. 기본 API (피드 조회, 스와이프 로깅)
@app.get("/api/v1/jobs/feed")
async def get_job_feed():
    try:
        query = supabase.table("jobs").select("*").order("created_at", desc=True).limit(20)
        result = query.execute()
        return {"jobs": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/jobs/swipe")
async def record_swipe(payload: SwipeActionRequest):
    try:
        swipe_data = {"user_id": payload.user_id, "job_id": payload.job_id, "action": payload.action}
        supabase.table("swipes").upsert(swipe_data).execute()
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 3. 자동 크롤링 & AI 파싱 파이프라인
async def run_crawler_pipeline():
    # 타겟 공고 원문 (실제 상용화 시 BeautifulSoup을 이용해 URL에서 동적으로 긁어옵니다)
    target_urls = ["https://example.com/careers/job-1234"]

    async with httpx.AsyncClient(timeout=10.0) as client:
        for url in target_urls:
            try:
                # 테스트용 모의 원문 데이터
                raw_text = """
                [채용 공고] 데이터 엔지니어 (Data Engineer) 대규모 채용
                담당 업무: 전사 데이터 파이프라인 구축, DW 설계 및 운영, Airflow 기반 배치 작업.
                자격 요건: Python, SQL 능숙자. 관련 경력 3년 이상 7년 이하.
                우대 사항: 대용량 트래픽 처리 경험, 클라우드(AWS/GCP) 환경 경험.
                근무지: 서울 판교. 연봉: 6,000만원 ~ 8,000만원.
                복리후생: 매월 체력단련비 지급, 유연근무제.
                """

                prompt = f"""
                채용 공고 원문을 분석하여 JSON 형태로 변환하세요.
                원문: {raw_text}
                반환 필수 필드 (JSON):
                - company_name: 기업명
                - position_title: 직무 제목
                - experience_min: 최소 요구 경력 (숫자만, 신입은 0)
                - experience_max: 최대 요구 경력 (숫자만)
                - location_short: 근무지 짧은 요약
                - salary_min: 최소 연봉 (단위: 만원)
                - salary_max: 최대 연봉 (단위: 만원)
                - summary_points: 매력 포인트 3가지 (각 1줄, 배열 형태)
                - details_tasks: 주요 업무 요약
                - details_reqs: 자격 요건 요약
                - details_perks: 혜택 및 복지 요약
                """

                res = ai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Return only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"}
                )

                parsed_data = json.loads(res.choices[0].message.content)

                db_payload = {
                    "company_name": parsed_data.get("company_name", "미상"),
                    "position_title": parsed_data.get("position_title", "직무 미상"),
                    "experience_min": parsed_data.get("experience_min", 0),
                    "experience_max": parsed_data.get("experience_max", 99),
                    "location_short": parsed_data.get("location_short", "위치 미상"),
                    "salary_min": parsed_data.get("salary_min", 0),
                    "salary_max": parsed_data.get("salary_max", 0),
                    "summary_points": parsed_data.get("summary_points", []),
                    "detail_content": {
                        "tasks": parsed_data.get("details_tasks", ""),
                        "reqs": parsed_data.get("details_reqs", ""),
                        "perks": parsed_data.get("details_perks", "")
                    },
                    "apply_url": url,
                    "is_active": True
                }

                supabase.table("jobs").insert(db_payload).execute()
                await asyncio.sleep(2)

            except Exception as e:
                print(f"Error crawling {url}: {e}")
                continue

@app.post("/api/v1/admin/trigger-crawl")
async def trigger_crawling(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_crawler_pipeline)
    return {"message": "크롤링 파이프라인이 실행되었습니다."}
