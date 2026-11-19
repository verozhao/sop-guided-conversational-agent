"""Offline understanding: regular expressions and keyword rules.

This engine needs no network and no API token. It is what the tests run on,
and it takes over for a turn whenever a model call fails. It is narrower than
a language model: it handles common phrasings, not arbitrary ones.
"""

from __future__ import annotations

import re
from datetime import date

from .data import ClaimsData
from .normalize import MONTH_ALT, MONTHS, find_dates, norm_claim_id
from .state import PROCESS_CASE, RESOLVE_INTENT, VERIFY_ID, Session
from .understanding import Understanding, normalize_identity

_I = re.I

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
POLICY_RE = re.compile(
    r"\bpol(?:icy)?[\s\-#:]*(?:number|no\.?|num|#)?[\s\-#:]*(?:is\s+)?(?:pol[\s\-]?)?(\d{3,8})\b", _I
)
CLAIM_ID_RE = re.compile(
    r"\bcl[\s\-]?(\d{3,8})\b|\bclaim\s*(?:number|no\.?|num|#|id)\s*(?:is\s+)?[:#]?\s*(?:cl[\s\-]?)?(\d{3,8})\b",
    _I,
)
SSN_MARK_RE = re.compile(r"\[SSN ending (\d{4})\]")
PHONE_RE = re.compile(r"(?<![\w\-])(\+?\d[\d\s().\-]{8,16}\d)(?![\w\-])")
LAST4_RE = re.compile(
    r"(?:\bssn\b|\bsocial(?:\s+security)?(?:\s+number)?|\bnational\s+id\b|\bid\b|"
    r"\blast\s+(?:four|4)(?:\s+digits)?|\bending(?:\s+in)?)\D{0,40}?(?<!\d)(\d{4})(?!\d)",
    _I,
)
DOB_TRIGGER_RE = re.compile(r"(dob|d\.o\.b|date\s+of\s+birth|birth\s*date|birthday|born)\W*(?:is|on|was|:)?\W*$", _I)

_NAME_STOP = {
    "the", "a", "an", "and", "or", "my", "i", "i'm", "im", "calling", "call", "phone", "email", "dob",
    "date", "born", "policy", "policyholder", "ssn", "social", "is", "was", "here", "not", "sorry", "just",
    "trying", "her", "his", "their", "son", "daughter", "mother", "father", "mom", "dad", "wife", "husband",
    "about", "regarding", "with", "from", "on", "at", "for", "so", "but", "ok", "okay", "yes", "no", "sure",
    "fine", "already", "still", "very", "really", "frustrated", "upset", "angry", "confused", "worried",
    "looking", "having", "going", "wondering", "hoping", "interested", "aware", "afraid", "unable", "able",
    "happy", "glad", "done", "good", "back", "new", "representative", "rep", "authorized", "customer",
    "member", "insured", "holder", "owner", "it", "this", "that", "you", "me", "we", "they", "he", "she",
    "who", "what", "why", "how", "when", "where", "which", "also", "too", "please", "thanks", "thank",
    "hi", "hello", "hey", "dear", "to", "of", "in", "out", "up", "if", "as", "by", "be", "am", "are",
}
_COMMON = _NAME_STOP | {
    "go", "away", "now", "tell", "give", "need", "want", "help", "wait", "stop", "come", "never", "mind",
    "nothing", "something", "whatever", "forget", "hold", "hang", "listen", "look", "see", "know", "think",
    "said", "told", "because", "down", "off", "over", "again", "later", "today", "tomorrow", "yesterday",
    "right", "wrong", "true", "false", "maybe", "perhaps", "claim", "claims", "denied", "insurance",
    "healthcare", "health", "dental", "auto", "car", "one", "two", "three", "first", "second", "third",
    "last", "all", "set", "bye", "goodbye", "great", "cool", "nice", "send", "skip", "do", "does", "did",
    "get", "got", "can", "could", "would", "should", "will", "have", "has", "had", "more", "less", "any",
    "some", "there", "then", "than", "them", "these", "those", "say", "ask", "use", "let", "make",
}

_WORD = re.compile(r"\s*([A-Za-z][A-Za-z'\-]*)([.,;!?:]?)")
_STRONG_NAME = re.compile(r"\b(?:full\s+)?name(?:\s+is|\s*[:=]|'s|\s+on\s+the\s+policy\s+is)\s+", _I)
_WEAK_NAME = re.compile(r"\b(?:i\s+am|i'm|this\s+is|it's|it\s+is)\s+", _I)
_RELATIVES = (
    r"mother|mom|mum|father|dad|wife|husband|son|daughter|partner|spouse|brother|sister|"
    r"grandmother|grandfather|grandma|grandpa|aunt|uncle|client|friend"
)
_OTHER_NAME = re.compile(
    rf"\bmy\s+(?:{_RELATIVES})(?:'s\s+name\s+is|\s+is|,)?\s+|\bon\s+behalf\s+of\s+(?:my\s+\w+,?\s+)?|"
    r"\b(?:her|his|their)\s+(?:full\s+)?name\s+is\s+|\bpolicyholder(?:'s\s+name)?\s+is\s+",
    _I,
)
REP_RE = re.compile(
    rf"\bon\s+behalf\s+of\b|\bcalling\s+(?:in\s+)?for\s+(?:my|our|a|the)\b|\bfor\s+my\s+(?:{_RELATIVES})\b|"
    rf"\bmy\s+(?:{_RELATIVES})(?:'s)?\b.{{0,60}}\b(?:claim|policy)\b|"
    r"\b(?:her|his|their)\s+(?:claim|policy|date\s+of\s+birth|dob|ssn|social|phone|email|name)\b|"
    rf"\bi(?:'m|\s+am)\s+(?:her|his|their)\s+(?:{_RELATIVES})\b|\bpower\s+of\s+attorney\b|\bcaregiver\b|"
    r"\bi(?:'m|\s+am)\s+(?:her|his|their)\s+(?:authori[sz]ed\s+)?(?:representative|rep|guardian|attorney)\b|"
    r"\bauthori[sz]ed\s+representative\b",
    _I,
)
POLICYHOLDER_RE = re.compile(
    r"\bi(?:'m|\s+am)\s+the\s+(?:policy\s?holder|insured|member|account\s+holder)\b|\bit(?:'s|\s+is)\s+my\s+(?:own\s+)?policy\b",
    _I,
)

DECLINE_TEMPLATES = [
    r"(?:won't|will\s+not|wont|refuse\s+to|don't\s+want\s+to|do\s+not\s+want\s+to|not\s+going\s+to|rather\s+not|"
    r"prefer\s+not\s+to|not\s+comfortable|can't|cannot|not)\s+(?:\w+\s+){{0,2}}?(?:giv|shar|provid|tell|disclos|say)\w*\b"
    r".{{0,30}}?\b(?:{field})\b",
    r"(?:don't|do\s+not|can't|cannot)\s+(?:have|know|remember|recall)\b.{{0,30}}?\b(?:{field})\b",
    r"\b(?:no|skip\s+(?:the|my))\s+(?:{field})\b",
]
DECLINE_FIELDS = [
    ("id_last4", r"ssn|social|national\s+id"),
    ("phone", r"phone|cell|mobile"),
    ("email", r"e-?mail"),
    ("dob", r"date\s+of\s+birth|dob|birthday|birth\s*date"),
    ("full_name", r"name"),
]
REFUSAL_RE = re.compile(
    r"already\s+(?:told|gave|said|explained|provided|verified|went\s+through)|"
    r"\bjust\s+(?:tell|give|answer|show)\s+me\b|"
    r"\b(?:not|won't|will\s+not|wont|refuse\s+to|don't\s+want\s+to|do\s+not\s+want\s+to|not\s+going\s+to|"
    r"shouldn't\s+have\s+to|should\s+not\s+have\s+to|don't\s+have\s+to|not\s+comfortable)\s+"
    r"(?:giv|shar|provid|tell|verif|answer|say|disclos)\w*|"
    r"none\s+of\s+your\s+business|\bskip\s+(?:the\s+|this\s+|all\s+)?(?:verification|identity|security|questions|step)|"
    r"without\s+(?:verif\w+|all\s+(?:this|these))|\bi\s+refuse\b|\bno\s+way\b|stop\s+asking|"
    r"\bcan(?:'t|not)?\s+you\s+just\b|\bi\s+said\s+no\b",
    _I,
)
PROCESS_Q_RE = re.compile(
    r"why\s+(?:do|would|should|must|does)\s+(?:you|i|we|she|he|they)\s+(?:even\s+|really\s+|still\s+)?"
    r"(?:need|have|ask|give|provide|verify|want|require)|"
    r"why\s+(?:is|does)\s+(?:this|that|verification|it|consent)\s+(?:needed|necessary|required|matter)|"
    r"why\s+(?:are\s+you\s+asking|all\s+this|so\s+many)|"
    r"what\s+(?:can|could|else\s+can|do|should)\s+i\s+(?:use|give|provide)|"
    r"what\s+(?:info|information|details|options|other\s+options|else)\s+(?:do\s+you\s+need|are\s+there|can\s+i)|"
    r"is\s+(?:this|it|that)\s+(?:secure|safe|confidential|private|really\s+necessary|necessary|required)|"
    r"are\s+you\s+(?:a\s+|an\s+)?(?:bot|robot|ai|human|real|machine|person|computer)|"
    r"who\s+am\s+i\s+(?:talking|speaking|chatting)\s+(?:to|with)|who\s+are\s+you|what\s+are\s+you|"
    r"what\s+can\s+you\s+(?:do|help)|how\s+long\s+(?:will|does)\s+(?:this|verification|it)\s+take|"
    r"what\s+happens\s+(?:next|after|then|to\s+my\s+(?:data|info))|do\s+i\s+have\s+to|"
    r"what\s+will\s+be\s+in\s+(?:the|that)\s+(?:email|summary)|what('s|\s+is)\s+in\s+the\s+(?:email|summary)|"
    r"which\s+(?:e-?mail|address)|what\s+(?:e-?mail|address)|where\s+will\s+(?:it|you)\s+(?:send|be\s+sent|go)|"
    r"what(?:'s|\s+is)\s+in\s+(?:it|that)|how\s+does\s+(?:this|that|it)\s+work|"
    r"\b(?:is\s+this|are\s+you|am\s+i\s+(?:talking|speaking|chatting)\s+(?:to|with))\b.{0,20}"
    r"\b(?:person|human|bot|robot|machine|ai|agent|computer)\b",
    _I,
)
WHAT_NEED_RE = re.compile(r"what\s+(?:do|does|else\s+do)\s+you\s+(?:still\s+|guys\s+)?need\b|what\s+(?:should|do)\s+i\s+(?:give|say|tell)\b", _I)
BOT_QUESTION_RE = re.compile(
    r"\b(?:is\s+this|are\s+you|am\s+i\s+(?:talking|speaking|chatting)\s+(?:to|with))\b.{0,20}"
    r"\b(?:person|human|bot|robot|machine|ai|agent|computer)\b",
    _I,
)

INTENT_RULES = [
    (
        "payment_or_amounts",
        r"\bhow\s+much\b|\bamounts?\b|reimburs|\bpaid\b|\bpayments?\b|\bpay\s?out\b|\bpay\b|\bmoney\b|\bowed?\b|"
        r"\bnet\s+(?:pay|fee)\b|\ballowed\b|\b(?:a|the|my)\s+check\b|\bcosts?\b",
    ),
    (
        "denial_explanation",
        r"\bwhy\b.*\b(?:denied|rejected|declined|turned\s+down)|\breason\b.*\b(?:denial|denied|rejected)|"
        r"\b(?:denied|rejected)\b.*\bwhy\b|explain.*\b(?:denial|denied)|what\s+happened\s+(?:to|with)\s+my\s+claim|"
        r"\bwhy\s+(?:was|is|did)\s+(?:it|that|this)\b|\bdenial\s+letter\b|"
        r"\bdenial\b.*\b(?:understand|explain|mean)|\b(?:understand|explain)\b.*\bdenial\b",
    ),
    (
        "document_requirements",
        r"\bdocuments?\b|\bpaperwork\b|\brecords?\b|what\s+do\s+(?:you|i)\s+need\s+to\s+(?:send|submit|provide|upload)|"
        r"what(?:'s|\s+is)\s+(?:missing|needed|required)|\bupload|\bsubmit|\bsend\s+(?:in|you|it|them)\b|\bfax\b|"
        r"\bmail\s+(?:it|them|in)\b|pathology|office\s+note|diagnosis\s+report|\bestimate\b|\bphotos?\b|\bscan|"
        r"\bpdf\b|file\s+(?:format|type)|\bportal\b",
    ),
    (
        "appeal_or_next_steps",
        r"\bappeal|\bnext\s+steps?\b|what\s+(?:do|should|can)\s+i\s+do\b|\bwhat\s+now\b|\bdeadline\b|"
        r"\bfix\s+(?:this|it)\b|reconsider|re-?open|\bdispute\b|\bcontest\b|\boverturn",
    ),
    (
        "claim_status",
        r"\bstatus\b|\bupdate\b|where\s+(?:is|are)\s+(?:my|the|we)\b|\bprogress\b|any\s+news|"
        r"been\s+(?:approved|processed|reviewed)|still\s+(?:open|pending|processing|in\s+progress)|"
        r"what(?:'s|\s+is)\s+(?:happening|going\s+on)\s+with",
    ),
    ("general_claim_question", r"\bclaims?\b|\bcase\b"),
]

TYPE_RULES = [
    ("healthcare", r"\bhealth\s?care\b|\bhealth\b|\bmedical\b|\bhospital\b"),
    ("dental", r"\bdental\b|\bdentist\b|\btooth\b|\bteeth\b"),
    ("auto", r"\bauto\b|\bcar\b|\bvehicle\b|\bcollision\b"),
]
STATUS_RULES = [
    ("denied", r"\bdenied\b|\bdenial\b|\brejected\b|\bturned\s+down\b"),
    ("open", r"\bopen\b|\bin\s+progress\b|\bpending\b|\bprocessing\b|\bongoing\b|\bunder\s+review\b"),
    ("closed", r"\bclosed\b|\bsettled\b|\bcompleted\b|\bpaid\s+out\b|\bresolved\b"),
]
MONTH_CTX_RE = re.compile(
    rf"\b(?:from|in|of|last|this|back\s+in|filed\s+in|since|around|during|for)\s+({MONTH_ALT})\b|"
    rf"\b({MONTH_ALT})\s+(?:claim|20\d{{2}})\b",
    _I,
)
YEAR_CTX_RE = re.compile(rf"\b(?:from|in|of|since|during)\s+(20\d{{2}})\b|\b{MONTH_ALT}\s+(20\d{{2}})\b", _I)

EMOTION_RULES = [
    (
        "angry",
        r"\b(?:ridiculous|unacceptable|outrageous|furious|pissed|angry|bullshit|stupid|useless|incompetent|"
        r"terrible|worst|wtf|damn|screw|scam|idiots?|absurd|insane|joke\s+of\s+a)\b|!!+",
    ),
    (
        "frustrated",
        r"frustrat|already\s+(?:told|gave|said|explained|provided)|how\s+many\s+times|waste\s+of\s+(?:my\s+)?time|"
        r"annoy|fed\s+up|sick\s+of|\bcome\s+on\b|\bseriously\b|keep\s+asking|not\s+helpful|going\s+in\s+circles|"
        r"just\s+tell\s+me|for\s+the\s+(?:third|second|last)\s+time|\bugh\b|\bi\s+said\b",
    ),
    (
        "anxious",
        r"worried|anxious|scared|nervous|afraid|stress|can't\s+afford|cannot\s+afford|panic|urgent|desperate|"
        r"don't\s+know\s+what\s+to\s+do|freaking\s+out|running\s+out\s+of\s+time|overwhelmed",
    ),
    ("sad", r"devastat|heartbroken|crying|in\s+tears|passed\s+away|\bdied\b|grief|so\s+sad|depress"),
    (
        "confused",
        r"confus|don't\s+understand|do\s+not\s+understand|what\s+do\s+you\s+mean|i'm\s+lost|makes\s+no\s+sense|"
        r"doesn't\s+make\s+sense|not\s+sure\s+what|\bhuh\b|unclear",
    ),
]

HUMAN_RE = re.compile(
    r"\b(?:speak|talk|chat|connect|transfer|put|get|escalate|switch)\b.{0,30}"
    r"\b(?:human|person|agent|representative|rep|supervisor|manager|someone|somebody|operator|specialist)\b|"
    r"\b(?:real|live|actual)\s+(?:person|human|agent)\b|\bhuman\b.{0,10}\bplease\b|"
    r"^\W*(?:a\s+|the\s+)?(?:human|agent|representative|operator|supervisor|person)(?:\s+please)?\W*$",
    _I,
)
HUMAN_NEG_RE = re.compile(r"(?:don't|do\s+not|no\s+need|not)\b.{0,15}\b(?:want|need)\b.{0,25}\b(?:human|person|agent|representative)", _I)

YES_RE = re.compile(
    r"^\W*(?:yes|yeah|yep|yup|sure|ok|okay|please\s+do|go\s+ahead|correct|right|that's\s+right|that\s+is\s+right|"
    r"that's\s+correct|absolutely|definitely|sounds\s+good|of\s+course|yes\s+please|alright|fine|do\s+it|"
    r"that's\s+the\s+one|that\s+one|exactly)\b",
    _I,
)
NO_RE = re.compile(
    r"^\W*(?:no|nope|nah|not\s+now|no\s+thanks|no\s+thank\s+you|negative|not\s+really|never\s?mind|"
    r"don't|do\s+not)\b",
    _I,
)
DONE_RE = re.compile(
    r"\b(?:that's\s+all|that\s+is\s+all|thats\s+all|that's\s+it|that\s+is\s+it|nothing\s+else|no\s+more\s+questions|"
    r"i'm\s+good|im\s+good|i\s+am\s+good|all\s+set|that's\s+everything|that\s+is\s+everything|that'll\s+be\s+all|"
    r"that\s+will\s+be\s+all|bye|goodbye|good\s+bye|have\s+a\s+(?:good|nice|great)\s+(?:day|one)|we're\s+done|"
    r"i'm\s+done|im\s+done|no\s+further\s+questions)\b",
    _I,
)
THANKS_ONLY_RE = re.compile(r"^\W*(?:ok(?:ay)?[,.\s]*)?(?:thanks|thank\s+you|ty|cheers|great|perfect|got\s+it)[\w\s,.!]{0,25}$", _I)

EMAIL_SKIP_RE = re.compile(
    r"\bskip\b|\bno\s+need\b|don't\s+(?:send|email|bother)|do\s+not\s+(?:send|email)|\bno\s+e-?mail\b|"
    r"not\s+necessary|without\s+(?:the\s+|an\s+)?e-?mail|\bpass\b",
    _I,
)
EMAIL_SEND_RE = re.compile(r"\bsend\b|\be-?mail\s+(?:it|me|that)\b|\bforward\b|\bgo\s+ahead\b", _I)
EMAIL_VOLUNTEER_RE = re.compile(
    r"\b(?:summary|recap)\b|\be-?mail\s+me\b|\bsend\s+me\s+an?\s+e-?mail\b|\banything\s+by\s+e-?mail\b|"
    r"\bno\s+e-?mails?\b|\bdon't\s+e-?mail\b",
    _I,
)

ALTERNATIVE_RE = re.compile(
    r"(?:don't|do\s+not|can't|cannot|couldn't|unable\s+to|no\s+longer|never)\s+(?:have|get|find|obtain|locate)|"
    r"\blost\s+(?:it|the|my)\b|\balternative|\binstead\s+of\b|\bsubstitute|\breplacement\b|"
    r"\bother\s+(?:proof|documents?)\b|not\s+available",
    _I,
)

DOMAIN_RE = re.compile(
    r"\b(?:claims?|policy|policies|insurance|insurer|coverage|covered|deductible|premium|appeal|documents?|"
    r"reimburs\w*|verif\w*|identity|account|adjuster|benefits?|denial|denied|provider|pathology|"
    r"representative|consent|summary)\b",
    _I,
)
OOS_RE = re.compile(
    r"\b(?:weather|joke|poem|recipe|song|lyrics|story|essay|python|javascript|coding|algorithm|"
    r"reinforcement\s+learning|rl|machine\s+learning|deep\s+learning|neural|llm|chatgpt|gpt|openai|president|"
    r"election|capital\s+of|stock|stocks|bitcoin|crypto|football|soccer|basketball|nba|nfl|movie|movies|netflix|"
    r"homework|math|equation|translate|meaning\s+of\s+life|horoscope|lottery|restaurant|flight|hotel|"
    r"recommend|medical\s+advice|legal\s+advice)\b|\bshould\s+i\s+(?:get|take|have|sue|see)\b|"
    r"^[\d\s+\-*/x().=?]*\d\s*[+\-*/x]\s*\d[\d\s+\-*/x().=?]*$|"
    r"\byour\s+(?:instructions|prompt|system\s+prompt|rules)\b|"
    r"\bignore\s+(?:all\s+|your\s+|the\s+|previous\s+)*(?:instructions|rules)\b",
    _I,
)
QUESTION_START_RE = re.compile(
    r"^\W*(?:what|who|when|where|why|how|which|can|could|would|will|do|does|did|is|are|tell|explain|"
    r"define|write|give|show|describe|compute|calculate|translate|name|list)\b",
    _I,
)
GENERAL_KNOWLEDGE_RE = re.compile(
    r"^\W*(?:what|who|where|when)(?:'s|\s+is|\s+are|\s+was|\s+were)\s+"
    r"(?!my\b|it\b|this\b|that\b|your\b|in\s+(?:it|that|the)\b|"
    r"(?:the\s+)?(?:status|deadline|reason|next|appeal|amount|claim|process|problem|issue|hold-?up|delay|"
    r"address|e-?mail|summary|difference|plan)\b)",
    _I,
)
GENERAL_TASK_RE = re.compile(
    r"^\W*(?:please\s+)?(?:write|tell\s+me\s+a|explain|define|calculate|compute|translate|sing|draw|summari[sz]e)\b"
    r"(?!\s+(?:it|that|this|why|how|what|my|the\s+(?:claim|denial|decision|reason|process|status|appeal)|me\s+(?:why|how|what))\b)",
    _I,
)
FOLLOW_ON_RE = re.compile(r"\b(?:the\s+)?other\s+(?:one|claim)\b|\banother\s+(?:one|claim)\b|\bthe\s+one\s+(?:from|in|for)\b|\ba\s+different\s+claim\b", _I)
OTHER_RE = re.compile(r"\bother\b|\banother\b|\bdifferent\b", _I)
ADDRESS_CHANGE_RE = re.compile(r"\b(?:different|another|other|new|work|personal|my\s+wife's|my\s+husband's)\s+(?:e-?mail|address)\b|\binstead\b", _I)

GREETING_RE = re.compile(
    r"^\W*(?:hi|hello|hey|good\s+(?:morning|afternoon|evening)|thanks|thank\s+you|ty|cheers|ok|okay|great|"
    r"got\s+it|i\s+see|alright|hmm+|sorry|help(?:\s+me)?(?:\s+please)?|i\s+need\s+(?:some\s+)?help|"
    r"(?:can|could|would)\s+you\s+help(?:\s+me)?)\b",
    _I,
)
ORDINAL_RE = re.compile(r"\b(first|1st|second|2nd|third|3rd|fourth|4th|last|latest|newest|most\s+recent|oldest)\b", _I)
_ORDINALS = {
    "first": 0, "1st": 0, "latest": 0, "newest": 0, "most recent": 0,
    "second": 1, "2nd": 1, "third": 2, "3rd": 2, "fourth": 3, "4th": 3, "last": -1, "oldest": -1,
}


def _take_name(rest: str, require_caps: bool) -> str:
    tokens: list[str] = []
    pos = 0
    while len(tokens) < 4:
        match = _WORD.match(rest, pos)
        if not match:
            break
        word, punct = match.group(1), match.group(2)
        possessive = word.lower().endswith("'s")
        if possessive:
            word = word[:-2]
        if not word or word.lower() in _NAME_STOP:
            break
        if require_caps and not word[0].isupper():
            break
        tokens.append(word)
        pos = match.end()
        if punct or possessive:
            break
    return " ".join(tokens)


def _first_name(pattern: re.Pattern[str], text: str, require_caps: bool) -> str:
    for match in pattern.finditer(text):
        name = _take_name(text[match.end():], require_caps)
        if name:
            return name
    return ""


def _mask(text: str, span: tuple[int, int]) -> str:
    return text[: span[0]] + " " * (span[1] - span[0]) + text[span[1]:]


def _bare_name(segment: str, lenient: bool) -> str:
    cleaned = re.sub(r"^\W*(?:full\s+)?(?:name)\W*", "", segment.strip(), flags=_I).strip(" .,;:!?")
    tokens = cleaned.split()
    if not 2 <= len(tokens) <= 4 or not all(re.fullmatch(r"[A-Za-z][A-Za-z'\-]*", t) for t in tokens):
        return ""
    if any(t.lower() in _COMMON for t in tokens):
        return ""
    if all(t[0].isupper() for t in tokens):
        return cleaned
    # Lower-case answers such as "margaret chen" are accepted only when they
    # cannot be a sentence: a wrongly read name would count as a failed attempt.
    if lenient and len(tokens) <= 3 and not any(re.search(r"(?:ing|tion|ment)$", t.lower()) for t in tokens):
        return cleaned
    return ""


def rule_understand(session: Session, text: str, data: ClaimsData, today: date) -> Understanding:
    u = Understanding(source="rules")
    low = text.lower()
    memory = session.memory
    in_verify = session.phase == VERIFY_ID
    pending = session.pending

    # ---- role -------------------------------------------------------------
    if REP_RE.search(text):
        u.caller_role = "representative"
    elif POLICYHOLDER_RE.search(text):
        u.caller_role = "policyholder"
    is_rep = u.caller_role == "representative" or (
        u.caller_role == "unknown" and memory.caller_role == "representative"
    )

    # ---- structured identity fields, masking each as it is found ----------
    raw: dict[str, str] = {}
    work = text

    match = EMAIL_RE.search(work)
    if match:
        raw["email"] = match.group(0)
        work = _mask(work, match.span())

    match = POLICY_RE.search(work)
    if match:
        raw["policy_number"] = match.group(1)
        work = _mask(work, match.span())

    hints: dict[str, object] = {}
    match = CLAIM_ID_RE.search(work)
    if match:
        hints["claim_id"] = norm_claim_id(match.group(1) or match.group(2))
        work = _mask(work, match.span())

    match = SSN_MARK_RE.search(work)
    if match:
        raw["id_last4"] = match.group(1)
        work = _mask(work, match.span())

    for year, month, day, span in find_dates(work):
        if year is None:
            continue
        before = work[max(0, span[0] - 30): span[0]]
        if DOB_TRIGGER_RE.search(before) or year <= today.year - 14:
            raw.setdefault("dob", f"{year:04d}-{month:02d}-{day:02d}")
        else:
            hints.setdefault("month", month)
            hints.setdefault("year", year)
        work = _mask(work, span)

    match = PHONE_RE.search(work)
    if match and 10 <= len(re.sub(r"\D", "", match.group(1))) <= 11:
        raw["phone"] = match.group(1)
        work = _mask(work, match.span())

    if "id_last4" not in raw:
        match = LAST4_RE.search(work)
        if match:
            raw["id_last4"] = match.group(1)
            work = _mask(work, match.span(1))

    # ---- names ------------------------------------------------------------
    letters = [ch for ch in text if ch.isalpha()]
    shouting = len(letters) > 12 and sum(ch.isupper() for ch in letters) / len(letters) > 0.7
    self_name = _first_name(_STRONG_NAME, text, False)
    if not self_name and not shouting:
        weak = _first_name(_WEAK_NAME, text, True)
        self_name = weak if len(weak.split()) >= 2 else ""
    if not self_name:
        match = re.search(r"\b(?:this\s+is|it's|i'm|i\s+am)\s+([a-z'\-]+\s+[a-z'\-]+)\s+(?:calling|speaking|here)\b", text, _I)
        if match and not any(tok.lower() in _COMMON for tok in match.group(1).split()):
            self_name = match.group(1)
    other_name = _first_name(_OTHER_NAME, text, True)
    if is_rep:
        if other_name:
            raw["full_name"] = other_name
        if self_name and self_name != other_name:
            u.representative_name = self_name
    elif self_name:
        raw["full_name"] = self_name

    # Short list-style answers during verification: "Jane Doe, 0000"
    if in_verify:
        for segment in re.split(r"[,;\n]|\band\b|[.!?](?:\s|$)", work):
            piece = segment.strip()
            if not piece:
                continue
            if "id_last4" not in raw and re.fullmatch(r"\W*(?:it'?s\s+|its\s+)?(\d{4})\W*", piece, _I):
                raw["id_last4"] = re.sub(r"\D", "", piece)
                continue
            sentence_like = bool(QUESTION_START_RE.match(low) or OOS_RE.search(low) or "?" in text)
            name = _bare_name(
                piece, lenient=pending in {"identity", "rep_name"} and len(text.split()) <= 6 and not sentence_like
            )
            if name and sentence_like and not name.istitle():
                name = ""
            if not name:
                continue
            if pending == "rep_name" or (is_rep and memory.identity.get("full_name") and not memory.representative_name):
                if not u.representative_name:
                    u.representative_name = name
            elif "full_name" not in raw and not (is_rep and u.representative_name == name):
                raw["full_name"] = name

    u.identity = normalize_identity(raw, today)
    if u.representative_name:
        u.representative_name = " ".join(part.capitalize() for part in u.representative_name.split())

    # ---- declined fields and refusal --------------------------------------
    for field_name, pattern in DECLINE_FIELDS:
        if field_name in u.identity:
            continue
        if any(re.search(template.format(field=pattern), low) for template in DECLINE_TEMPLATES):
            u.declined_fields.append(field_name)
    if "full_name" in u.declined_fields and len(u.declined_fields) > 1:
        u.declined_fields.remove("full_name")  # "name" is too generic when another field is named
    u.process_question = bool(PROCESS_Q_RE.search(low))
    identity_given = any(k in u.identity for k in ("full_name", "dob", "phone", "email", "id_last4"))
    if REFUSAL_RE.search(low) and not identity_given and not u.declined_fields:
        u.refuses_verification = True

    # ---- intent, topics, case hints ---------------------------------------
    for intent, pattern in INTENT_RULES:
        if re.search(pattern, low):
            u.intent = intent
            break
    if session.phase == PROCESS_CASE and re.fullmatch(r"\W*(?:but\s+)?why(?:\s+not|\s+is\s+that|\s+though)?\W*", low):
        u.intent = "denial_explanation"
    if WHAT_NEED_RE.search(low):
        if in_verify:
            u.process_question = True
        elif u.intent == "none":
            u.intent = "document_requirements"
    if u.process_question and u.intent == "general_claim_question":
        u.intent = "none"

    topics: list[str] = []
    for row in data.guidelines.get("claim_followup_guidance", []):
        if any(re.search(rf"\b{re.escape(phrase)}", low) for phrase in row.get("match_any", [])):
            topics.append(row["topic"])
    if re.search(r"\b(?:how|where)\s+(?:do|can|should|would)\s+i\s+(?:send|submit|upload)\b|\bfax\b|\bmail\s+(?:it|them)\b", low):
        topics.append("submission_method")
    if re.search(r"how\s+(?:will|do|would)\s+i\s+know\b|did\s+you\s+(?:get|receive)\b|confirm\w*\b.{0,30}\b(?:receiv|got|get)", low):
        topics.append("receipt_confirmation")
    if ALTERNATIVE_RE.search(low) and u.intent in {"document_requirements", "appeal_or_next_steps", "none"}:
        if not u.declined_fields and (session.phase != VERIFY_ID or re.search(r"report|note|document|estimate|photo", low)):
            topics.append("missing_required_material_alternatives")
    u.followup_topics = list(dict.fromkeys(topics))
    if u.followup_topics and u.intent in {"none", "general_claim_question"} and session.phase in {PROCESS_CASE, RESOLVE_INTENT}:
        u.intent = "document_requirements"

    claim_context = (
        bool(re.search(r"\bclaims?\b|\bcase\b", low))
        or pending == "choose_case"
        or session.phase == RESOLVE_INTENT
        or "claim_id" in hints
        or (session.verified and bool(FOLLOW_ON_RE.search(low)))
    )
    hint_text = work.lower()
    if claim_context:
        for claim_type, pattern in TYPE_RULES:
            if re.search(pattern, hint_text):
                hints["claim_type"] = claim_type
                break
        for status, pattern in STATUS_RULES:
            if re.search(pattern, hint_text):
                hints["status"] = status
                break
        match = MONTH_CTX_RE.search(hint_text)
        if match:
            hints.setdefault("month", MONTHS[(match.group(1) or match.group(2)).lower()])
        match = YEAR_CTX_RE.search(hint_text)
        if match:
            hints.setdefault("year", int(match.group(1) or match.group(2)))
        elif re.search(r"\blast\s+year\b", hint_text):
            hints.setdefault("year", today.year - 1)
        elif re.search(r"\bthis\s+year\b", hint_text):
            hints.setdefault("year", today.year)
    if claim_context and session.case_id and OTHER_RE.search(hint_text) and "claim_id" not in hints:
        hints["exclude_current"] = True
    u.case_hints = hints

    if pending == "choose_case" and session.last_candidates and "claim_id" not in hints:
        match = ORDINAL_RE.search(low) or re.fullmatch(r"\W*(?:number\s*|#)?([1-9])\W*", low)
        if match:
            key = re.sub(r"\s+", " ", match.group(1).lower())
            index = _ORDINALS.get(key, int(key) - 1 if key.isdigit() else None)
            if index is not None and -1 <= index < len(session.last_candidates):
                u.selected_case_id = session.last_candidates[index]

    if u.intent != "none" or u.followup_topics:
        u.question = text.strip()[:300]

    # ---- emotion, human, yes/no, done, email ------------------------------
    for emotion, pattern in EMOTION_RULES:
        if re.search(pattern, low):
            u.emotion = emotion
            break
    if u.emotion == "neutral" and shouting:
        u.emotion = "angry"

    u.wants_human = (
        bool(HUMAN_RE.search(text))
        and not HUMAN_NEG_RE.search(text)
        and not is_rep_statement(low)
        and not BOT_QUESTION_RE.search(low)
    )

    if YES_RE.match(low):
        u.affirmation = "yes"
    elif NO_RE.match(low):
        u.affirmation = "no"

    u.done = bool(DONE_RE.search(low))
    if pending == "anything_else" and u.intent == "none" and not u.followup_topics:
        if u.affirmation == "no" or THANKS_ONLY_RE.match(low):
            u.done = True

    if pending == "email_offer" and (ADDRESS_CHANGE_RE.search(low) or u.process_question):
        u.process_question = True
    elif pending == "email_offer":
        if EMAIL_SKIP_RE.search(low) or u.affirmation == "no":
            u.email_summary_choice = "skip"
        elif EMAIL_SEND_RE.search(low) or u.affirmation == "yes":
            u.email_summary_choice = "send"
    elif EMAIL_VOLUNTEER_RE.search(low) and "email" not in u.identity:
        if EMAIL_SKIP_RE.search(low) or re.search(r"\bno\s+e-?mails?\b|don't\s+e-?mail", low):
            u.email_summary_choice = "skip"
        elif re.search(r"\b(?:send|e-?mail|mail|get|have|want|like)\b", low) and not u.process_question:
            u.email_summary_choice = "send"
    if u.email_summary_choice != "none" and u.intent in {"document_requirements", "general_claim_question"}:
        if not re.search(r"\bdocuments?\b|\bclaims?\b", low):
            u.intent = "none"
            u.question = ""

    # ---- scope ------------------------------------------------------------
    domain_hit = bool(
        u.identity or u.representative_name or u.intent != "none" or u.case_hints or u.selected_case_id
        or u.process_question or u.wants_human or u.email_summary_choice != "none" or u.done
        or u.refuses_verification or u.declined_fields or DOMAIN_RE.search(low)
    )
    short_answer = u.affirmation != "none" or bool(GREETING_RE.match(low))
    oos_marker = bool(OOS_RE.search(low))
    general = bool(GENERAL_KNOWLEDGE_RE.match(low) or GENERAL_TASK_RE.match(low))
    question_like = "?" in text or bool(QUESTION_START_RE.match(low))
    if (oos_marker or general) and not domain_hit:
        u.scope = "out_of_scope"
    elif oos_marker:
        u.scope = "mixed"
    elif not domain_hit and not short_answer and question_like and not session.verified:
        u.scope = "out_of_scope"  # no context yet: an unrelated question
    if u.scope == "out_of_scope":
        u.affirmation = "none"
    elif question_like and not u.question and not short_answer and not u.process_question and not u.identity:
        u.question = text.strip()[:300]  # in scope, but no rule knows how to answer it
    return u


def is_rep_statement(low: str) -> bool:
    """'I am her representative' states a role. It is not a request for a human."""
    return bool(re.search(r"\b(?:i'm|i\s+am|as)\s+(?:her|his|their|the|an?)\s+(?:authori[sz]ed\s+)?representative\b", low))
