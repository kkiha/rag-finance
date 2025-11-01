from __future__ import annotations
import re
from typing import List, Tuple, Optional

# 네가 쓰던 목록(필요시 보강/수정)
COMPANY_LIST = [
    '가온칩스', 'KMH하이텍', 'LG이노텍', 'LG전자', 'LX세미콘',
    'SFA반도체', 'SK하이닉스', '고영', '네오와인', '네패스', '네패스아크',
    '넥스트칩', '동진쎄미켐', '라닉스', '리노공업', '비트리',
    '삼성전자', '세미파이브', '신성이엔지', '아나패스', '어보브', '에이디테크놀로지',
    '에프에스티', '엘오티베큠', '와이아이케이', '와이팜', '유진테크', '이오테크닉스',
    '자람테크놀로지', '제주반도체', '젬백스', '주성엔지니어링', '지니틱스',
    '코아시아넥셀', '테크윙', '텔레칩스', '티씨케이', '티엘아이',
    '피델릭스', '피에스케이홀딩스', '하나머티리얼즈'
]

COMPANY_CODE = [
    '399720', '052900', '011070', '066570', '108320',
    '036540', '000660', '098460', '085670', '033640', '330860',
    '396270', '005290', '317120', '058470', '138690',
    '005930', '330170', '011930', '123860', '102120', '200710',
    '036810', '083310', '232140', '332370', '084370', '039030',
    '389020', '080220', '082270', '036930', '303030',
    '089890', '089030', '054450', '064760', '062860',
    '032580', '031980', '166090'
]

NAME_TO_CODE = {n: c for n, c in zip(COMPANY_LIST, COMPANY_CODE) if c != '000000'}
CODE_TO_NAME = {c: n for n, c in NAME_TO_CODE.items()}

def extract_company_from_query(query_text: str) -> Tuple[str, str]:
    name = max([c for c in COMPANY_LIST if c in query_text], key=len, default="")
    code: Optional[str] = None
    m = re.search(r'(?<!\d)(\d{6})(?!\d)', query_text)
    if m:
        code = m.group(1)
    if name and not code:
        code = NAME_TO_CODE.get(name, "")
    if code and not name:
        name = CODE_TO_NAME.get(code, "")
    return (name or "", code or "")


def extract_company_name_from_text(text: str, company_list: Optional[List[str]] = None) -> str:
    """본문에서 등장하는 기업명 중 가장 긴 항목을 선택."""
    if company_list is None:
        company_list = COMPANY_LIST
    hits = [c for c in company_list if c in (text or "")]
    return max(hits, key=len) if hits else ""


def extract_company_code_from_text(text: str) -> Optional[str]:
    """본문에서 종목코드(6자리 숫자)를 추출. 매핑 목록 우선."""
    codes = re.findall(r'(?<!\d)(\d{6})(?!\d)', text or "")
    for code in codes:
        if code in CODE_TO_NAME:
            return code
    return codes[0] if codes else None


def resolve_company_from_text(text: str) -> Tuple[str, str]:
    """본문 전체를 기반으로 (기업명, 종목코드) 추정."""
    name = extract_company_name_from_text(text)
    code = extract_company_code_from_text(text or "")

    if name and not code:
        code = NAME_TO_CODE.get(name, "")
    if code and not name:
        name = CODE_TO_NAME.get(code, "")

    return (name or "", code or "")
