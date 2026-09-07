import os
import json
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from supabase import create_client, Client
from openai import OpenAI

# FastAPI 앱 초기화
app = FastAPI(title="Job Swipe App AI Onboarding API", version="1.0.0")

# -------------------------------------------------------------
# 1. 환경 변수 및 클라이언트 설정
# (Render 배포 시 환경변수 창에서 입력한 키 값이 자동으로 연결됩니다)
# -------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
ai_client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------------------------------------------
# 2. 요청/응답 데이터 모델 정의
# -------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str  # 'user' 또는 'assistant'
    content: str

class OnboardingExtractRequest(BaseModel):
    dialogue_history: List[ChatMessage]

class ParsedProfile(BaseModel):
    primary_role: str = Field(description="정제된 표준 직무명 (예: HR, 물류/SCM, 기획 등)")
    years_of_experience: int = Field(description="총 경력 연차 (숫자만 추출, 신입은 0)")
    min_salary: int = Field(description="희망 최소 연봉 (단위: 만 원, 미언급 시 0)")
    preferred_locations: List[str] = Field(description="희망 근무 지역 목록 (시/구 단위)")
    core_competencies: List[str] = Field(description="보유 핵심 역량 및 스킬 키워드 3~5개")
    summary_raw: str = Field(description="구직자의 전체 프로필을 1줄로 요약한 문장")

class SwipeActionRequest(BaseModel):
    user_id: str
    job_id: str
    action: str  # 'PASS' 또는 'APPLY'

# -------------------------------------------------------------
# 3. API 엔드포인트
# -------------------------------------------------------------

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "Job Swipe Matching Backend"}

@app.post("/api/v1/onboarding/parse-and-save")
async def parse_and_save_profile(payload: OnboardingExtractRequest):
    """
    온보딩 대화 기록을 AI가 분석하여 정형 데이터로 변환 후 Supabase DB에 저장
    """
    try:
        dialogue_text = "\n".join([f"{msg.role}: {msg.content}" for msg in payload.dialogue_history])

        system_prompt = (
            "당신은 채용 플랫폼의 전문 커리어 어드바이저입니다. "
            "구직자와의 온보딩 대화 기록을 분석하여 정확한 스펙을 JSON 포맷으로 추출하세요. "
            "연봉은 '만 원' 단위의 숫자여야 하며, 명시되지 않은 경우 0으로 표기합니다. "
            "경력 연차는 명시된 숫자를 기준으로 정수값으로 반환하세요."
        )

        response = ai_client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"다음 대화에서 프로필 데이터를 추출해줘:\n\n{dialogue_text}"}
            ],
            response_format=ParsedProfile
        )

        extracted: ParsedProfile = response.choices[0].message.parsed

        db_data = {
            "primary_role": extracted.primary_role,
            "years_of_experience": extracted.years_of_experience,
            "min_salary": extracted.min_salary,
            "preferred_locations": extracted.preferred_locations,
            "core_competencies": extracted.core_competencies,
            "summary_raw": extracted.summary_raw
        }

        db_response = supabase.table("profiles").insert(db_data).execute()

        if not db_response.data:
            raise HTTPException(status_code=500, detail="데이터베이스 저장 실패")

        saved_profile = db_response.data[0]
        return {
            "success": True,
            "profile_id": saved_profile["id"],
            "profile": db_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/jobs/feed")
async def get_job_feed(user_id: Optional[str] = None):
    """
    카드 스와이프 화면용 공고 20건 반환
    """
    try:
        query = supabase.table("jobs").select("*").eq("is_active", True).limit(20)
        result = query.execute()
        return {"jobs": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/jobs/swipe")
async def record_swipe(payload: SwipeActionRequest):
    """
    왼쪽(PASS) 또는 오른쪽(APPLY) 스와이프 액션 DB 기록
    """
    if payload.action not in ["PASS", "APPLY"]:
        raise HTTPException(status_code=400, detail="action은 'PASS' 또는 'APPLY'여야 합니다.")

    try:
        swipe_data = {
            "user_id": payload.user_id,
            "job_id": payload.job_id,
            "action": payload.action
        }
        res = supabase.table("swipes").upsert(swipe_data).execute()
        return {"success": True, "data": res.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))