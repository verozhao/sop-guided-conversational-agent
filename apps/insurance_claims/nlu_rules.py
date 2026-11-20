"""Offline understanding: regular expressions and keyword rules.

This engine needs no network and no API token. It is what the tests run on,
and it takes over for a turn whenever a model call fails. It is narrower than
a language model, and it is built to fail safe: anything with a consequence
(sending the email, a transfer to a person, switching claims, counting an
identity attempt, ending the call) is read only from unambiguous wording.
When a message is unclear the reading stays empty and the policy asks again.
"""

from __future__ import annotations

import re
from datetime import date

from .data import ClaimsData
from .normalize import MONTH_ALT, MONTHS, find_dates, norm_claim_id
from .state import CLOSED, PROCESS_CASE, RESOLVE_INTENT, VERIFY_ID, Session
from .understanding import MENTIONS_OTHER_RE, Understanding, clean_representative_name, normalize_identity

_I = re.I

# --------------------------------------------------------------------------- identity values

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
# "my husband's birthday is", "my doctor's number is": the value after it is somebody else's
_THIRD_PARTY_RE = re.compile(
    r"\bmy\s+(?:mother|mom|mum|father|dad|wife|husband|son|daughter|partner|spouse|brother|sister|friend|kid|child|"
    r"doctor|dentist|lawyer|attorney|boss|employer|provider|pharmacy|clinic|hospital)(?:'s)?\s+(?:[\w']+\s+){0,4}$",
    re.I,
)
# "call me back at ..." gives a number to ring, which need not be the one on the policy. "reach me at ..." is a detail.
_CALLBACK_RE = re.compile(r"\b(?:call|text|ring|phone)\s+(?:me\s+)?back\s+(?:at|on)\s*$|\bcall-?\s?back\s+number\s+(?:is\s+)?$", re.I)
DOB_TRIGGER_RE = re.compile(r"(dob|d\.o\.b|date\s+of\s+birth|birth\s*date|birthday|born)\W*(?:is|on|was|:)?\W*$", _I)

# --------------------------------------------------------------------------- names

# Words that end a name after "my name is" / "this is".
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
# Words that are never part of a name typed on its own. A short reply is only taken
# for a name when none of its words is in here.
_COMMON = _NAME_STOP | {
    "go", "away", "now", "tell", "give", "need", "want", "help", "wait", "stop", "come", "never", "mind",
    "nothing", "something", "whatever", "forget", "hold", "hang", "listen", "look", "see", "know", "think",
    "said", "told", "because", "down", "off", "over", "again", "later", "today", "tomorrow", "yesterday",
    "right", "wrong", "true", "false", "maybe", "perhaps", "claim", "claims", "denied", "insurance",
    "healthcare", "health", "dental", "auto", "car", "one", "two", "three", "first", "second", "third",
    "last", "all", "set", "bye", "goodbye", "great", "cool", "nice", "send", "skip", "do", "does", "did",
    "get", "got", "can", "could", "would", "should", "will", "have", "has", "had", "more", "less", "any",
    "some", "there", "then", "than", "them", "these", "those", "say", "ask", "use", "let", "make",
} | set(
    """id number numbers cell mobile makes sense uh huh um umm hmm oh ah aha wow boy gosh geez jeez dude god
    lol seriously though speak english spanish fair enough its thats whats hows lets dont cant wont im ive
    ill youre address anything anyone everyone everything nobody always actually alright another anyway
    bad bit both busy check clear close closed correct deal details different each easy end even ever every
    exactly far fast few file found four five free full give given gone guess hard having hello hope info
    information issue keep kind least leave left like listen lot many matter mean might mine minute moment
    most much must name names next nope not number often once only open other our own people person pretty
    probably problem question questions quite rather ready real really same sec seems show simple since
    slow so soon sort stuff such take talk thing things time total trouble try under understand until us
    very way week well were while whole whose with without wondering work worse yeah yep yup yet your yours
    human agent bot robot machine operator supervisor manager someone somebody please thanks cheers
    idk dunno nvm huh? what? fuck shit damn hell crap screw bloody stupid ridiculous useless""".split()
)

_WORD = re.compile(r"\s*([A-Za-z][A-Za-z'\-]*)([.,;!?:]?)")
_CONTRACTION = re.compile(r"'(?:s|m|re|ll|ve|d|t)$", _I)
_STRONG_NAME = re.compile(r"\b(?:full\s+)?name(?:\s+is|\s*[:=]|'s|\s+on\s+the\s+policy\s+is)\s+", _I)
_WEAK_NAME = re.compile(r"\b(?:i\s+am|i'm|this\s+is|it's|it\s+is|that's|that\s+is)\s+", _I)
_SUFFIX_NAME = re.compile(r"\b(?:this\s+is|it's|i'm|i\s+am)\s+([a-z'\-]+\s+[a-z'\-]+)\s+(?:calling|speaking|here)\b", _I)

_RELATIVES = (
    r"mother|mom|mum|father|dad|wife|husband|son|daughter|partner|spouse|brother|sister|"
    r"grandmother|grandfather|grandma|grandpa|aunt|uncle|client|friend|neighbou?r|patient"
)
_ROLES = rf"(?:{_RELATIVES}|lawyer|attorney|representative|caregiver|carer|guardian|assistant|broker|accountant)"
# The same words as a set: "my friend Jane Doe" names Jane Doe, not "Friend Jane Doe".
_KIN = set(
    """mother mom mum father dad wife husband son daughter partner spouse brother sister grandmother grandfather
    grandma grandpa aunt uncle client friend neighbor neighbour patient lawyer attorney representative caregiver
    carer guardian assistant broker accountant""".split()
)

_NAME_RUN = r"((?:[A-Za-z][A-Za-z\-]+\s+){0,3}[A-Za-z][A-Za-z\-]+)"
_SELF = r"(?:i\s+am|i'm|im|this\s+is|it's|it\s+is)"
# "I'm Jane Doe's husband", "Jane Doe's son here": the name is the policyholder's, the caller is somebody else.
_POSSESSIVE_ROLE = re.compile(
    rf"\b{_SELF}\s+{_NAME_RUN}'s\s+{_ROLES}\b|^\W*{_NAME_RUN}'s\s+{_ROLES}\s+(?:here|calling|speaking)\b", _I
)
# "calling for Jane Doe" in any letter case. "calling about ..." needs capitals, because a topic can follow it.
_CALLING_FOR = re.compile(rf"\b(?:calling|phoning|ringing|asking)\s+(?:in\s+)?(?:for|on\s+behalf\s+of)\s+{_NAME_RUN}", _I)
_CALLING_ABOUT = re.compile(r"\b[Cc]alling\s+(?:in\s+)?(?:about|regarding)\s+((?:[A-Z][A-Za-z\-]+\s+){1,2}[A-Z][A-Za-z\-]+)")
# "I'm the caregiver for Jane Doe", "Jane Doe is my mother"
_ROLE_FOR = re.compile(rf"\b(?:{_SELF}|as)\s+(?:(?:the|a|an|her|his|their|your)\s+)?{_ROLES}\s+(?:for|of|to)\s+{_NAME_RUN}", _I)
_IS_MY = re.compile(rf"\b{_NAME_RUN}\s+is\s+my\s+(?:{_RELATIVES})\b", _I)
_OTHER_NAME = re.compile(
    rf"\bmy\s+(?:{_RELATIVES})(?:'s\s+name\s+is|\s+is|,)?\s+|\bon\s+behalf\s+of\s+(?:my\s+\w+,?\s+)?|"
    rf"\b(?:her|his|their)\s+(?:full\s+)?name\s+is\s+|\bpolicyholder(?:'s\s+name)?\s+is\s+|\b{_ROLES}\s+of\s+",
    _I,
)

# The caller is acting for the policyholder. Strong cues say so outright.
REP_STRONG_RE = re.compile(
    rf"\bon\s+behalf\s+of\b|\bcalling\s+(?:in\s+)?for\s+(?:my|our|a|the|her|him|them)\b|"
    rf"\bfor\s+my\s+(?:{_RELATIVES})\b|\bi(?:'m|\s+am)\s+(?:her|his|their)\s+{_ROLES}\b|"
    rf"\bi(?:'m|\s+am)\s+the\s+{_ROLES}\s+of\b|\bpower\s+of\s+attorney\b|\bauthori[sz]ed\s+representative\b|"
    r"\bi(?:'m|\s+am)\s+not\s+the\s+policy\s?holder\b|\b(?:her|his|their)\s+behalf\b|"
    rf"\bthe\s+policy\s?holder'?s\s+{_ROLES}\b|\bthe\s+policy\s?holder\s+is\s+my\s+(?:{_RELATIVES})\b|"
    rf"\b(?:this\s+is|it's)\s+(?:her|his|their)\s+{_ROLES}\b|^\W*(?:her|his|their)\s+{_ROLES}\b|"
    r"\bnot\s+mine\b|\bnot\s+my\s+(?:own\s+)?(?:policy|claim|account)\b",
    _I,
)
# Weak cues: talking about somebody else's claim or details.
REP_WEAK_RE = re.compile(
    rf"\bmy\s+(?:{_RELATIVES})'s\s+(?:\w+\s+){{0,3}}?(?:claims?|policy|account|insurance)\b|"
    rf"\bmy\s+(?:{_RELATIVES})\s+(?:(?:has|had|got|received|filed)\s+an?\b|is\s+the\s+(?:policy\s?holder|insured|member))|"
    r"\b(?:her|his)\s+(?:full\s+)?(?:name|date\s+of\s+birth|dob|birthday|ssn|social|phone(?:\s+number)?|e-?mail)\b|"
    r"\btheir\s+(?:date\s+of\s+birth|dob|birthday|ssn|social)\b|"
    r"\b(?:about|regarding|on|for|with|check)\s+(?:her|his)\s+(?:\w+\s+){0,2}?(?:claims?|policy)\b",
    _I,
)
# Once the gate has opened, only an explicit statement about oneself counts.
SELF_REP_RE = re.compile(
    rf"\bi(?:'m|\s+am)\s+(?:actually\s+|really\s+)?(?:her|his|their)\s+{_ROLES}\b|\bi(?:'m|\s+am)\s+not\s+the\s+policy\s?holder\b|"
    r"\bcalling\s+on\s+(?:her|his|their)\s+behalf\b|"
    rf"\bi(?:'m|\s+am)\s+(?:actually\s+|really\s+|only\s+|just\s+)?calling\s+(?:for|on\s+behalf\s+of)\s+(?:her|him|them|my\s+(?:{_RELATIVES}))\b|"
    rf"\bthis\s+is\s+(?:actually\s+)?(?:her|his|their)\s+{_ROLES}\b|\bnot\s+my\s+(?:own\s+)?(?:policy|claim)\b",
    _I,
)
# Answers to "are you the policyholder yourself?". A no is looked for first, and a yes
# only counts when nothing in the message negates it: "No, I am not" must never read as "I am".
ITS_ME_RE = re.compile(
    r"\b(?:that's|it's|this\s+is|that\s+is|it\s+is)\s+me\b|\bmyself\b|\bmy\s+own\b|^\W*(?:yes[\s,]*)?i\s+am\W*$|"
    r"\bi(?:'m|\s+am)\s+the\s+policy\s?holder\b|^\W*speaking\W*$",
    _I,
)
NOT_ME_RE = re.compile(
    r"\bnot\s+me\b|\b(?:i'm|i\s+am)\s+not\b|\bnot\s+the\s+policy\s?holder\b|\bnot\s+mine\b|\bsomeone\s+else\b|"
    r"\bfor\s+(?:her|him|them|someone)\b|\bon\s+(?:her|his|their)\s+behalf\b",
    _I,
)
POLICYHOLDER_RE = re.compile(
    r"\bi(?:'m|\s+am)\s+the\s+(?:policy\s?holder|insured|member|account\s+holder)(?!'s)\b|"
    r"\b(?:it(?:'s|\s+is)|this\s+is)\s+my\s+(?:own\s+)?policy\b|\bmy\s+own\s+(?:policy|claim)\b",
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
    r"\bcan(?:'t|not)?\s+you\s+just\b|\bi\s+said\s+no\b|\bi(?:'d|\s+would)\s+rather\s+not\b|"
    r"\bnot\s+doing\s+(?:this|that)\b|(?:\bnot|n't)\s+(?:feel\s+)?comfortable\b|\bstill\s+no\b|\bnot\s+happening\b|"
    r"^\W*(?:let's\s+)?skip\s+(?:it|this|that)\b|\byou\s+don't\s+need\s+(?:that|this|my|any)\b|\bno\s+chance\b",
    _I,
)
PLAIN_NO_RE = re.compile(r"^\W*(?:no|nope|nah|no\s+thanks|no\s+thank\s+you|no\s+way|never|i\s+won't|i\s+will\s+not)\W*$", _I)

# --------------------------------------------------------------------------- questions about the call

PROCESS_Q_RE = re.compile(
    r"why\s+(?:do|would|should|must|does)\s+(?:you|i|we|she|he|they)\s+(?:even\s+|really\s+|still\s+)?"
    r"(?:need|have|ask|give|provide|verify|want|require|trust)|"
    r"why\s+(?:is|does)\s+(?:this|that|verification|it|consent)\s+(?:needed|necessary|required|matter)|"
    r"why\s+(?:are\s+you\s+asking|all\s+this|so\s+many)|"
    r"what\s+(?:can|could|else\s+can|do|should)\s+i\s+(?:use|give|provide)|"
    r"what\s+(?:info|information|details|options|other\s+options|else)\s+(?:do\s+you\s+need|are\s+there|can\s+i)|"
    r"is\s+(?:this|it|that)\s+(?:even\s+)?(?:secure|safe|confidential|private|really\s+necessary|necessary|required)|"
    r"who\s+am\s+i\s+(?:talking|speaking|chatting)\s+(?:to|with)|who\s+are\s+you|what\s+are\s+you|"
    r"what\s+can\s+you\s+(?:do|help)|how\s+long\s+(?:will|does)\s+(?:this|verification|it)\s+take|"
    r"what\s+happens\s+(?:next|after|then|to\s+my\s+(?:data|info))|do\s+i\s+have\s+to|"
    r"what\s+will\s+be\s+in\s+(?:the|that|it)|what('s|\s+is)\s+in\s+(?:the\s+(?:e-?mail|summary)|it|that)|"
    r"which\s+(?:e-?mail|address)|what\s+(?:e-?mail|address)|where\s+will\s+(?:it|you)\s+(?:send|be\s+sent|go)|"
    r"how\s+does\s+(?:this|that|it)\s+work",
    _I,
)
BOT_QUESTION_RE = re.compile(
    r"\b(?:is\s+this|are\s+you|am\s+i\s+(?:talking|speaking|chatting)\s+(?:to|with))\b.{0,20}"
    r"\b(?:person|human|bot|robot|machine|ai|agent|computer)\b",
    _I,
)
# Before verification: clarifications about the step the caller is being asked to do.
VERIFY_Q_RE = re.compile(
    r"\bwhat\s+do\s+you\s+mean\b|\b(?:repeat|rephrase|say\s+(?:that|it)\s+again|come\s+again)\b|"
    r"\bwhich\s+(?:three|two|ones?|details?|info|information|of\s+(?:these|those))\b|\bhow\s+many\b|"
    r"\b(?:can|could|may)\s+i\s+(?:use|give|just|skip)\b|\bwhat\s+(?:if|happens\s+if)\b|\bwhat\s+else\b|\bwhat\s+for\b|"
    r"\bis\s+(?:my|a|an|the)\s+[\w'\s\-]{1,30}?\s+(?:ok|okay|fine|enough|accepted|acceptable|allowed|good)\b|"
    r"\bwill\s+(?:you|this|it)\s+(?:store|keep|save|share|sell|record|be\s+(?:stored|shared|recorded|saved))\b|"
    r"\bare\s+you\s+(?:still\s+)?(?:there|listening|recording)\b|\bdo\s+you\s+(?:need|want)\b|\bwhat\s+do\s+you\s+need\b|"
    r"\b(?:driver'?s?\s+licen[sc]e|passport|maiden\s+name|middle\s+name|zip\s+code|home\s+address|member\s+id)\b|"
    r"\b(?:instead|other\s+way|another\s+way|different\s+way|any\s+other|something\s+else|not\s+sure\s+what)\b|"
    r"\bwhy\b|\b(?:safe|secure|trust|privacy|private|scam|legit)\b|\bthis\s+for\b|\bthe\s+point\b|\bhow\s+come\b|"
    r"\breason\b|\boptions?\b|\blater\b|\banother\s+time\b|\bcall\s+back\b|\bcome\s+back\b|\bright\s+now\b|"
    r"\bdo\s+this\b|\bevery\s+time\b|\bhear\s+me\b|\banyone\s+there\b|\bhow\s+long\b|\benough\b|\bwho\s+is\s+this\b",
    _I,
)
SHORT_CLARIFY_RE = re.compile(
    r"^\W*(?:(?:sorry|um+|uh+|ok|okay|and|so|but|wait)[\s,]+)*"
    r"(?:what|how|how\s+so|huh|pardon|excuse\s+me|really|then\s+what|what\s+now|now\s+what|what\s+next|and|so|"
    r"meaning|which|which\s+ones?|how\s+come)\W*$",
    _I,
)
WAIT_RE = re.compile(
    r"\b(?:give\s+me|just|one|wait)\s+(?:a\s+)?(?:minute|moment|sec|second)\b|\b(?:hold|hang)\s+on\b|\bbear\s+with\s+me\b|"
    r"\blet\s+me\s+(?:check|find|look|grab|get)\b",
    _I,
)
WHAT_NEED_RE = re.compile(r"what\s+(?:do|does|else\s+do)\s+you\s+(?:still\s+|guys\s+)?need\b|what\s+(?:should|do)\s+i\s+(?:give|say|tell)\b", _I)

# --------------------------------------------------------------------------- claims

INTENT_RULES = [
    (
        "payment_or_amounts",
        r"\bhow\s+much\b|\bamounts?\b|reimburs|\bpaid\b|\bpayments?\b|\bpay\s?out\b|\bpay\b|\bmoney\b|\bowed?\b|"
        r"\bnet\s+(?:pay|fee)\b|\ballowed\b|\b(?:a|the|my)\s+check\b|\bcosts?\b",
    ),
    (
        "denial_explanation",
        r"\bwhy\b.*\b(?:deny|denied|reject|rejected|declined|turned\s+down)|\breason\b.*\b(?:denial|denied|rejected)|"
        r"\b(?:denied|rejected)\b.*\bwhy\b|explain.*\b(?:denial|denied)|what\s+happened\s+(?:to|with)\s+my\s+claim|"
        r"\bwhy\s+(?:was|is|did)\s+(?:it|that|this)\b|\bdenial\s+letter\b|"
        r"\bdenial\b.*\b(?:understand|explain|mean)|\b(?:understand|explain)\b.*\bdenial\b",
    ),
    (
        "document_requirements",
        r"\bdocuments?\b|\bpaperwork\b|\brecords?\b|what\s+do\s+(?:you|i)\s+need\s+to\s+(?:send|submit|provide|upload)|"
        r"what(?:'s|\s+is)\s+(?:missing|needed|required)|\bupload|\bsubmit|\bsend\s+(?:in|you|it|them)\b|\bfax\b|"
        r"(?<![e\-])\bmail\s+(?:it|them|in)\b|pathology|office\s+note|diagnosis\s+report|\bestimate\b|\bphotos?\b|\bscan|"
        r"\bpdf\b|file\s+(?:format|type)|\bportal\b",
    ),
    (
        "appeal_or_next_steps",
        r"\bappeal|\bnext\s+steps?\b|what\s+(?:do|should|can)\s+i\s+do\b|\bwhat\s+now\b|\bdeadline\b|"
        r"\bfix\s+(?:this|it)\b|reconsider|re-?open|\bdispute\b|\bcontest\b|\boverturn|\bmy\s+options\b|"
        r"\banything\s+(?:else\s+)?i\s+can\s+do\b",
    ),
    (
        "claim_status",
        r"\bstatus\b|\bupdate\b|where\s+(?:is|are)\s+(?:my|the|we)\b|\bprogress\b|any\s+news|"
        r"(?:been|got|get|gets)\s+(?:approved|processed|reviewed)|still\s+(?:open|pending|processing|in\s+progress)|"
        r"what(?:'s|\s+is)\s+(?:happening|going\s+on)\s+with|"
        r"when\s+(?:was|did)\s+(?:it|this|that|i|you|we|the\s+claim|my\s+claim)\s+(?:\w+\s+)?(?:filed?|submit(?:ted)?|opened|received)|"
        r"\bfiling\s+date\b|\bdate\s+(?:it\s+was\s+)?filed\b",
    ),
]
GENERAL_CLAIM_RE = re.compile(r"\bclaims?\b|\bcase\b", _I)
# Things a caller asks the assistant to do that it cannot: these get an honest "not from here".
ACTION_REQUEST_RE = re.compile(
    r"\b(?:approve|expedite|speed\s+up|rush|fast-?track|waive|cancel|withdraw|pay\s+me|"
    r"(?:file|open|start|make)\s+a\s+(?:new\s+)?claim|"
    r"(?:update|change|correct|fix)\s+my\s+(?:phone|address|e-?mail|name|number|details|bank|contact)|"
    r"send\s+me\s+(?:a\s+|the\s+|my\s+)?(?:check|payment|money))\b",
    _I,
)
# With a claim in focus, disagreeing with the outcome is a question about what can be done.
DISPUTE_RE = re.compile(
    r"\b(?:this|that|it)(?:'s|\s+is)\s+(?:wrong|a\s+mistake|not\s+right|incorrect|unfair|not\s+fair)\b|"
    r"\bcan't\s+be\s+right\b|\bi\s+disagree\b|\bthat's\s+a\s+mistake\b|\bmade\s+a\s+mistake\b",
    _I,
)
# Wording that is about the email itself. A documents question does not hide in it.
EMAILISH_RE = re.compile(
    r"\b(?:send|forward)\s+(?:it|that|me|one|the\s+(?:summary|e-?mail|recap))\b|\be-?mail\s+(?:it|me|that)\b|"
    r"\b(?:don't|do\s+not|not|never)\s+send\b",
    _I,
)

_TYPE_WORDS = {
    "healthcare": r"health\s?care|health|medical|hospital",
    "dental": r"dental|dentist|tooth|teeth",
    "auto": r"auto|car|vehicle|collision",
}
STATUS_RULES = [
    ("denied", r"\bdenied\b|\bdenial\b|\brejected\b|\bturned\s+down\b"),
    ("open", r"\bopen\b|\bin\s+progress\b|\bpending\b|\bprocessing\b|\bongoing\b|\bunder\s+review\b"),
    ("closed", r"\bclosed\b|\bsettled\b|\bcompleted\b|\bpaid\s+out\b|\bresolved\b"),
]
_REF = r"(?:claims?|case|one)"
MONTH_CTX_RE = re.compile(
    rf"\b(?:from|in|of|last|this|back\s+in|filed\s+in|since|around|during|for)\s+({MONTH_ALT})\b|"
    rf"\b({MONTH_ALT})\s+(?:claim|20\d{{2}})\b",
    _I,
)
YEAR_CTX_RE = re.compile(rf"\b(?:from|in|of|since|during)\s+(20\d{{2}})\b|\b{MONTH_ALT}\s+(20\d{{2}})\b", _I)
# With a claim in focus, a hint counts only when it is attached to a claim reference.
MONTH_REF_RE = re.compile(
    rf"\b{_REF}\s+(?:from|in|of|for|filed\s+in|back\s+in)\s+(?:last\s+)?({MONTH_ALT})\b(?:\s+(20\d{{2}}))?|"
    rf"\b({MONTH_ALT})(?:\s+(20\d{{2}}))?\s+{_REF}\b",
    _I,
)
YEAR_REF_RE = re.compile(
    rf"\b{_REF}\s+(?:from|in|of|for)\s+(20\d{{2}}|last\s+year|this\s+year)\b|\b(20\d{{2}}|last\s+year's|this\s+year's)\s+{_REF}\b",
    _I,
)
STATUS_REF_RE = re.compile(rf"\b(denied|rejected|open|pending|closed|settled)\s+(?:\w+\s+)?{_REF}\b", _I)
# When the agent has just listed claims, a month, a year or "the newer one" on its own is an answer.
BARE_MONTH_RE = re.compile(rf"\b({MONTH_ALT})\b", _I)
BARE_YEAR_RE = re.compile(r"\b(20\d{2})\b")
FOLLOW_ON_RE = re.compile(
    r"\b(?:an)?other\s+(?:\w+\s+)?(?:one|claims?)\b|\ba\s+different\s+(?:one|claim)\b|\b(?:second|next)\s+claim\b", _I
)
_ORDINAL_WORDS = r"first|1st|second|2nd|third|3rd|fourth|4th|last|latest|newest|newer|most\s+recent|more\s+recent|oldest|older|earlier|earliest"
ORDINAL_RE = re.compile(
    rf"\b({_ORDINAL_WORDS})\s+(?:one|claim|option|in\s+the\s+list)\b|"
    rf"^\W*(?:(?:the|um+|uh+|ok|okay|yes|yeah)\s+)*({_ORDINAL_WORDS})(?:\s+please)?\W*$",
    _I,
)
_ORDINALS = {
    "first": 0, "1st": 0, "latest": 0, "newest": 0, "newer": 0, "most recent": 0, "more recent": 0,
    "second": 1, "2nd": 1, "third": 2, "3rd": 2, "fourth": 3, "4th": 3,
    "last": -1, "oldest": -1, "older": -1, "earlier": -1, "earliest": -1,
}
# "I can't get the report": the thing that is missing has to be a document, or "it" / "them".
ALTERNATIVE_RE = re.compile(
    r"(?:don't|do\s+not|can't|cannot|couldn't|unable\s+to|no\s+longer|never)\s+(?:have|get|find|obtain|locate)\s+"
    r"(?:(?:a|an|the|my|that|this|those|these|any|either|both)\s+)?(?:\w+\s+){0,2}?"
    r"(?:reports?|notes?|records?|documents?|paperwork|estimates?|photos?|pictures?|cop(?:y|ies)|files?|forms?|"
    r"it|them|that|those|one|either|both)\b|"
    r"\blost\s+(?:it|the|my)\b|\balternative|\binstead\s+of\b|\bsubstitute|\breplacement\b|"
    r"\bother\s+(?:proof|documents?)\b|not\s+available",
    _I,
)

# --------------------------------------------------------------------------- feelings and requests

EMOTION_RULES = [
    (
        "angry",
        r"\b(?:ridiculous|unacceptable|outrageous|furious|pissed|angry|bullshit|stupid|useless|incompetent|"
        r"terrible|awful|horrible|worst|wtf|damn|screw|scam|idiots?|absurd|insane|nonsense|garbage|"
        r"a\s+joke|complaint|unbelievable|pathetic|disgrace(?:ful)?)\b|!!+",
    ),
    (
        "frustrated",
        r"frustrat|already\s+(?:told|gave|said|explained|provided)|how\s+many\s+times|waste\s+of\s+(?:my\s+)?time|"
        r"wasting\s+my|annoy|fed\s+up|sick\s+of|tired\s+of|\bcome\s+on\b|\bseriously\b|keep\s+asking|not\s+helpful|"
        r"going\s+in\s+circles|getting\s+nowhere|just\s+tell\s+me|for\s+the\s+(?:third|second|last)\s+time|\bugh\b|"
        r"\bi\s+said\b|can't\s+believe|not\s+listening|what\s+a\s+mess|hopeless|pointless|don't\s+have\s+time",
    ),
    (
        "anxious",
        r"worried|anxious|scared|nervous|afraid|stress|can't\s+afford|cannot\s+afford|panic|urgent|desperate|"
        r"don't\s+know\s+what\s+to\s+do|freaking\s+out|running\s+out\s+of\s+time|overwhelmed|can't\s+sleep|"
        r"bills?\s+(?:are\s+)?piling",
    ),
    ("sad", r"devastat|heartbroken|crying|in\s+tears|passed\s+away|\bdied\b|grief|so\s+sad|depress"),
    (
        "confused",
        r"confus|don't\s+understand|do\s+not\s+understand|what\s+do\s+you\s+mean|i'm\s+lost|makes\s+no\s+sense|"
        r"doesn't\s+make\s+sense|not\s+sure\s+what|(?<!uh\s)\bhuh\b|unclear|what\s+does\s+(?:that|this|it)\s+mean|"
        r"don't\s+(?:get|follow)|not\s+following|lost\s+me|in\s+plain\s+(?:english|words)|\bsimpler\b|"
        r"explain\s+(?:that|this|it)\s+again",
    ),
]

_PERSON = r"(?:human|person|agent|representative|rep|supervisor|manager|operator)"
_ANYONE = rf"(?:(?:a|an|the|your)\s+)?(?:real\s+|live\s+|actual\s+|human\s+)?(?:{_PERSON}|someone|somebody)"
HUMAN_RE = re.compile(
    "|".join(
        [
            # a request frame, then talking to somebody on the support side
            rf"\b(?:i\s+(?:want|need|wanna)|i(?:'d|\s+would)\s+(?:like|rather|prefer)|can\s+i|could\s+i|may\s+i|let\s+me|"
            rf"please|get\s+me|give\s+me|just)\b[^.?!]{{0,30}}?\b(?:speak|talk|chat)\s+(?:to|with)\s+{_ANYONE}\b",
            # asking to have a person
            rf"\b(?:i\s+(?:want|need)|i(?:'d|\s+would)\s+(?:like|rather\s+have|prefer)|can\s+i\s+(?:have|get)|give\s+me|get\s+me)\s+"
            rf"(?:to\s+)?(?:a|an|the|your)?\s*(?:real\s+|live\s+|actual\s+)?{_PERSON}\b",
            r"\b(?:transfer|connect)\s+me\b|\bput\s+me\s+through\b|\bescalate\s+(?:me|this|it|my)\b",
            rf"\b(?:speak|talk)\s+(?:to|with)\s+(?:a|an|your)\s+(?:real\s+|live\s+|actual\s+)?{_PERSON}\b",
            r"\bsomeone\s+(?:else\s+)?i\s+can\s+(?:talk|speak)\s+(?:to|with)\b",
            rf"\b(?:can|could)\s+(?:a|an)\s+(?:real\s+|live\s+|actual\s+)?{_PERSON}\s+(?:help|look|review|do|call)\b",
            rf"^\W*(?:a\s+|the\s+)?(?:real\s+|live\s+)?{_PERSON}(?:\s+please)?\W*$",
            r"\bhuman\b.{0,10}\bplease\b",
        ]
    ),
    _I,
)
# A negation cancels the request only when it governs the transfer or the person:
# "don't transfer me", "I'd rather not talk to a person". "I don't want a bot" cancels nothing.
HUMAN_NEG_RE = re.compile(
    rf"\b(?:don't|do\s+not|dont|not|never|no\s+need(?:\s+to)?|rather\s+not|without|won't)\s+"
    rf"(?:(?:want|wanna|need|have|to|be|you|just|really|please|even|ever|get|going|like)\s+){{0,3}}?"
    rf"(?:transfer|connect|escalat\w+|put\s+me|(?:talk|speak|chat)\w*\s+(?:to|with)\s+(?:(?:a|an|the|your)\s+)?"
    rf"(?:real\s+|live\s+|actual\s+)?(?:{_PERSON}|someone|somebody)|(?:want|need)\s+(?:(?:a|an)\s+)?{_PERSON})\b",
    _I,
)
HUMAN_ASKING_RE = re.compile(
    r"\b(?:do|should|must|will|would)\s+i\s+(?:really\s+)?(?:need|have)\s+to\b|\bshould\s+i\b|\bis\s+it\s+(?:better|worth)\b", _I
)
REP_STATEMENT_RE = re.compile(
    r"\b(?:i'm|i\s+am|as)\s+(?:her|his|their|the|an?)\s+(?:authori[sz]ed\s+)?representative\b|"
    r"\b(?:my|as\s+a)\s+representative\b",
    _I,
)

_YES = (
    r"(?:yes|yeah|yep|yup|uh\s+huh|mhm+|mm+\s*hmm+|sure|ok|okay|alright|fine|correct|right|absolutely|definitely|of\s+course|please\s+do|"
    r"go\s+ahead|sounds\s+good|do\s+it|that\s+works|that's\s+right|that\s+is\s+right|that's\s+correct|"
    r"that's\s+the\s+one|that\s+one|exactly|that\s+would\s+be\s+(?:great|good|nice|helpful|lovely)|"
    r"i(?:'d|\s+would)\s+like\s+that|go\s+for\s+it|why\s+not|i\s+suppose(?:\s+so)?|i\s+guess(?:\s+so)?)"
)
_HANDOFF_WORDS = set("connect transfer human person representative agent someone them through put with him her up over".split())
# Words that may follow a yes at the email offer without making it anything but a yes.
_YES_FILLER = set(
    """please thanks thank you do it that this send go ahead sure yes yeah yep ok okay great sounds good perfect
    would be i'd i like love the a summary email e-mail me to my address on file absolutely definitely of course
    fine alright right works lovely nice helpful appreciate very so much for now then just one""".split()
)
_NO = r"(?:no|nope|nah|negative|not\s+now|not\s+really|never\s?mind)"
YES_START_RE = re.compile(rf"^\W*{_YES}\b", _I)
NO_START_RE = re.compile(rf"^\W*{_NO}\b", _I)
YES_THEN_NO_RE = re.compile(rf"^\W*{_YES}[\s,.!]*(?:not|no|nope|nah)\b", _I)
NEGATION_RE = re.compile(r"\b(?:no|not|never|nothing|none|nope|nah)\b|n't\b", _I)
# After a leading "no", these mean the caller is not finished or is not simply declining.
NO_BUT_RE = re.compile(
    r"\b(?:wait|hold\s+on|actually|but|still|one\s+more|another|questions?|send|yes|skip|listening|wrong|"
    r"understand|confused|can't\s+be|hang\s+up|close|why|how|what|mean|meant|other|different|instead|not\s+that|"
    r"the\s+\w+\s+(?:one|claim))\b|\?|!!",
    _I,
)
MORE_RE = re.compile(
    r"\b(?:another|one\s+more|one\s+other|a\s+few\s+more|some\s+more|a\s+couple\s+more)\s+(?:questions?|things?)\b|"
    r"\bi\s+(?:still\s+)?ha(?:ve|d)\s+(?:a\s+|another\s+|more\s+|some\s+|a\s+few\s+)?questions?\b|"
    r"^\W*(?:(?:no|oh|um+|uh+|ok|okay|actually|sorry|hey)[\s,]+)*(?:wait|hold\s+on|hang\s+on)\b|"
    r"\b(?:don't|do\s+not)\s+(?:hang\s+up|close|end\s+(?:this|the)|go\s+yet)\b|\bnot\s+(?:done|finished)\b",
    _I,
)
DONE_RE = re.compile(
    r"\b(?:that's\s+all|that\s+is\s+all|thats\s+all|that's\s+it|that\s+is\s+it|nothing\s+else|no\s+more\s+questions|"
    r"no\s+further\s+questions|i'm\s+good|im\s+good|i\s+am\s+good|i'm\s+fine|all\s+set|all\s+good|"
    r"that's\s+everything|that\s+is\s+everything|that'll\s+be\s+all|that\s+will\s+be\s+all|we're\s+done|"
    r"i'm\s+done|im\s+done|good\s?bye|bye(?:\s+bye)?|have\s+a\s+(?:good|nice|great)\s+(?:day|one|evening|night|weekend)|"
    r"take\s+care|never\s?mind|(?:don't|do\s+not)\s+have\s+(?:any|any\s+other|any\s+more|any\s+further|more|other)\s+questions|"
    r"(?:don't|do\s+not)\s+have\s+anything\s+else|no\s+other\s+questions|nothing\s+(?:more|further)|"
    r"(?:need|have|got|gotta)\s+to\s+go|gotta\s+go|talk\s+(?:to\s+you\s+)?later|thanks\s+anyway|"
    r"i'll\s+call\s+(?:you\s+)?back|call\s+back\s+(?:later|another\s+time|tomorrow)|i'll\s+come\s+back|maybe\s+later)\b",
    _I,
)
# Words that may stand around a closing phrase without changing what it means.
_CLOSING_FILLER = set(
    """no nope nah ok okay great perfect alright thanks thank you u so much very well i think guess that helps
    helped helpful for now today then really a lot appreciate it your the help and yes yeah yep cool good got
    oh sure fine right just all im i'm we we're are am is was have nice day take care everything needed need
    wanted to know me again this time cheers ty lovely wonderful awesome excellent done with my questions
    honestly suppose believe sir ma'am mate later see ya ill i'll call back""".split()
)
PURE_NO_RE = re.compile(
    r"^\W*(?:no|nope|nah|no\s+thanks|no\s+thank\s+you|not\s+really|nothing|not\s+right\s+now|not\s+now)"
    r"(?:[\s,.!]*(?:thanks|thank\s+you))?\W*$",
    _I,
)
THANKS_ONLY_RE = re.compile(
    r"^\W*(?:(?:ok|okay|great|perfect|alright|got\s+it|cool)[\s,.!]*)*(?:thanks|thank\s+you|ty|cheers)"
    r"(?:[\s,.!]*(?:so\s+much|very\s+much|a\s+lot|that\s+helps|that\s+helped|that's\s+helpful|appreciate\s+it|"
    r"for\s+(?:your|the)\s+help))*\W*$",
    _I,
)

# The email summary
EMAIL_NEGATED_RE = re.compile(
    r"\b(?:don't|do\s+not|dont|not|never|no\s+need\s+to|won't|shouldn't)\s+(?:\w+\s+){0,2}?(?:send|e-?mail|forward|bother)\b|"
    r"\b(?:don't|do\s+not|dont)\s+(?:want|need)\b|\bno\s+e-?mails?\b|\bno\s+summary\b|\bskip\b|\bno\s+need\b|"
    r"\bnot\s+necessary\b|\b(?:i'll|i\s+will|let's|we'll|hard)\s+pass\b|^\W*pass\W*$|"
    r"\b(?:don't|do\s+not)\s+(?:have|use)\s+(?:an?\s+)?e-?mail\b",
    _I,
)
EMAIL_SEND_RE = re.compile(r"\bsend\b|\be-?mail\s+(?:it|me|that)\b|\bforward\b|\bgo\s+ahead\b", _I)
EMAIL_REQUEST_RE = re.compile(
    r"^\W*(?:(?:yes|yeah|sure|ok|okay)[\s,]+)?(?:can|could|would|will)\s+you\s+(?:please\s+)?(?:just\s+)?(?:send|e-?mail|forward)\b|"
    r"^\W*(?:can|could|may)\s+i\s+(?:please\s+)?(?:still\s+)?(?:get|have)\b[^?]{0,25}\b(?:summary|recap)\b",
    _I,
)
# "I'd like the summary after all" asks for it without the word send.
EMAIL_WISH_RE = re.compile(r"\b(?:want|like|need|get|have|take)\b[^.?!]{0,25}\b(?:summary|recap)\b", _I)
NOT_SKIP_RE = re.compile(r"\b(?:don't|do\s+not|dont|not)\s+skip(?:\s+(?:it|that))?\b", _I)
ADDRESS_CHANGE_RE = re.compile(
    r"\b(?:different|another|other|new|work|personal)\s+(?:e-?mail|address)\b|\binstead\b|\bthat\s+address\b|"
    r"\bnot\s+to\s+that\b",
    _I,
)
EMAIL_VOLUNTEER_RE = re.compile(
    r"\b(?:summary|recap)\b|\be-?mail\s+me\b|\bsend\s+me\s+an?\s+e-?mail\b|\banything\s+by\s+e-?mail\b|"
    r"\bno\s+e-?mails?\b|\bdon't\s+e-?mail\b",
    _I,
)
EMAIL_WORDS_RE = re.compile(r"\be-?mails?\b|\bsummary\b|\brecap\b|\bsend\s+(?:it|that)\b", _I)
# After closing, only wording that can mean nothing but the summary reopens the choice.
CLOSED_SEND_RE = re.compile(r"\bsummary\b|\brecap\b|\b(?:send|e-?mail|forward)\s+(?:it|that)\b", _I)

# --------------------------------------------------------------------------- scope

DOMAIN_RE = re.compile(
    r"\b(?:claims?|policy|policies|policyholder|insurance|insurer|coverage|covered|deductible|premium|appeal|"
    r"documents?|reimburs\w*|verif\w*|identity|account|adjuster|benefits?|denial|denied|provider|pathology|"
    r"representative|consent|summary|deadline)\b",
    _I,
)
# Softer signs that a question belongs to claims support, used only when nothing else matched.
CLAIMISH_RE = re.compile(
    r"\b(?:bill(?:s|ed|ing)?|invoice|receipt|statement|letter|notice|hospital|clinic|doctor|dentist|surgeon|surgery|"
    r"procedure|treatment|prescription|pharmacy|mri|x-?ray|lab|accident|collision|repair|review(?:ed|er|ing)?|"
    r"decision|approv\w*|reject\w*|den(?:y|ies)|cover(?:s|ing)?|pay(?:ing|ments?)?|paid|owe[ds]?|charge[ds]?|refund|"
    r"copay|co-pay|out\s+of\s+pocket|in-?network|plan|member|expedite|rush|urgent|complain\w*|submit\w*|upload\w*|"
    r"hear\s+back|call(?:ed)?\s+(?:me\s+)?back|contact\s+(?:me|you)|healthcare|health\s?care|medical|dental|auto|"
    r"vehicle|car|forms?|(?:phone|contact|fax)\s+number|i\s+can\s+call|customer\s+service|office|in\s+person|"
    r"human|person|agent|supervisor|manager|transfer(?:red)?|service|diagnos\w+|codes?|coinsurance|filed|filing|"
    r"date\s+of|visit|appointment|provider|facility)\b",
    _I,
)
# Short follow-ups that only make sense as a reaction to what was just said.
CLARIFY_RE = re.compile(
    r"\b(?:are\s+you\s+sure|you\s+sure|for\s+real|really|seriously|how\s+so|how\s+come|since\s+when|why\s+not|"
    r"and\s+then|then\s+what|what\s+else|what\s+do\s+you\s+mean|explain|understand|say\s+again|pardon|excuse\s+me|"
    r"going\s+on|happening|what(?:'s|\s+is)\s+wrong|the\s+(?:problem|matter|issue|hold-?up|delay)|taking\s+so\s+long)\b",
    _I,
)
# Words that tie a short message to the conversation so far.
ANAPHORA_RE = re.compile(
    r"\b(?:it|this|that|these|those|they|them|one|same|you\s+(?:paid|sent|need|said|have|got|received|mean)|"
    r"approve|expedite|reopen|now|next|yet|still|nothing|come\s+through|approved)\b",
    _I,
)
OOS_RE = re.compile(
    r"\b(?:weather|poem|recipe|song|lyrics|essay|python|javascript|coding|algorithm|"
    r"reinforcement\s+learning|rl|machine\s+learning|deep\s+learning|neural|llm|chatgpt|gpt|openai|president|"
    r"election|capital\s+of|stocks?|bitcoin|crypto|football|soccer|basketball|nba|nfl|world\s+cup|super\s+bowl|"
    r"movies?|netflix|homework|math|equation|translate|meaning\s+of\s+life|horoscope|lottery|restaurants?|"
    r"flights?|hotels?|vacation|taxes|recommend|medical\s+advice|legal\s+advice|rain(?:ing)?|snow(?:ing)?|"
    r"forecast|temperature|aliens?|music|cover\s+letter|flat\s+tire)\b|"
    r"\btell\s+me\s+a\s+(?:joke|story)\b|\bjoke\s+about\b|\bshould\s+i\s+(?:get|take|have|sue|see)\b|"
    r"\byour\s+(?:instructions|prompt|system\s+prompt|rules)\b|"
    r"\bignore\s+(?:all\s+|your\s+|the\s+|previous\s+)*(?:instructions|rules)\b",
    _I,
)
# Arithmetic needs a plus, a times or an equals sign. Digits with slashes or dashes are a date or a phone number.
MATH_RE = re.compile(r"^[\d\s+\-*/x().]*\d\s*(?:[+*x]\s*\d[\d\s+\-*/x().]*[=?\s]*|[\-/]\s*\d[\d\s+\-*/x().]*=[=?\s]*)$", _I)
GENERAL_KNOWLEDGE_RE = re.compile(
    r"^\W*(?:what|who|where|when)(?:'s|\s+is|\s+are|\s+was|\s+were)\s+"
    r"(?!my\b|it\b|this\b|that\b|your\b|in\s+(?:it|that|the)\b|"
    r"(?:the\s+|a\s+)?(?:status|deadline|reason|next|appeal|amount|claim|process|problem|issue|hold-?up|delay|"
    r"address|e-?mail|summary|difference|plan|policyholder|policy|point|catch|alternative|matter)\b|"
    r"going\s+on\b|happening\b|wrong\b|taking\b)|"
    r"^\W*who\s+(?:won|invented|wrote|discovered|painted)\b|"
    r"^\W*how\s+(?:do|can|would)\s+(?:i|you|we)\s+(?:bake|cook|make|build|code|program|learn|play|draw|lose\s+weight)\b|"
    r"^\W*how\s+(?:tall|old|far|big|hot|cold)\s+(?:is|are|was)\b|^\W*why\s+is\s+the\s+(?:sky|sea|grass|ocean)\b",
    _I,
)
GENERAL_TASK_RE = re.compile(
    r"^\W*(?:please\s+)?(?:write|explain|define|calculate|compute|translate|sing|draw|summari[sz]e|tell\s+me\s+about)\b"
    r"(?!\s+(?:it|that|this|why|how|what|my|the\s+(?:claim|denial|decision|reason|process|status|appeal)|me\s+(?:why|how|what))\b)",
    _I,
)
QUESTION_START_RE = re.compile(
    r"^\W*(?:what|who|when|where|why|how|which|can|could|would|will|do(?!\s+not\b)|does|did|is|are|tell|explain|"
    r"define|write|give(?!\s+me\s+a\s+(?:minute|moment|sec|second)\b)|show|describe|compute|calculate|translate|name|list)\b",
    _I,
)
GREETING_RE = re.compile(
    r"^\W*(?:hi|hello|hey|good\s+(?:morning|afternoon|evening|day)|thanks|thank\s+you|ty|cheers|ok|okay|great|"
    r"got\s+it|i\s+see|alright|hmm+|sorry|help(?:\s+me)?(?:\s+please)?|i\s+need\s+(?:some\s+)?help|"
    r"(?:can|could|would)\s+you\s+help(?:\s+me)?|how\s+are\s+you|how's\s+it\s+going|how\s+is\s+it\s+going|"
    r"what's\s+up|how\s+do\s+you\s+do|nice\s+to\s+meet)\b",
    _I,
)


# --------------------------------------------------------------------------- helpers


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


def _name_tail(candidate: str) -> str:
    """The part of a run of words that can be a name ("hi jane doe about her" -> "jane doe").

    One word alone only counts when it is capitalised, because a lower-case word is as
    likely to be a topic ("calling for reimbursement") as a name.
    """
    tokens = candidate.split()
    while tokens and (tokens[0].lower() in _COMMON or tokens[0].lower() in _KIN):
        tokens.pop(0)
    name: list[str] = []
    for token in tokens:
        if token.lower() in _COMMON or token.lower() in _KIN or len(name) == 3:
            break
        name.append(token)
    if len(name) >= 2 or (len(name) == 1 and name[0][0].isupper()):
        return " ".join(name)
    return ""


def _mask(text: str, span: tuple[int, int]) -> str:
    return text[: span[0]] + " " * (span[1] - span[0]) + text[span[1]:]


def _bare_name(segment: str, allow_lower: bool) -> tuple[str, bool]:
    """A reply that is nothing but a name, and whether it is only a guess (typed in lower case)."""
    cleaned = re.sub(r"^\W*(?:full\s+)?(?:name)\W*", "", segment.strip(), flags=_I).strip(" .,;:!?")
    tokens = cleaned.split()
    if not 2 <= len(tokens) <= 4 or not all(re.fullmatch(r"[A-Za-z][A-Za-z'\-]*", t) for t in tokens):
        return "", False
    if any(t.lower() in _COMMON or _CONTRACTION.search(t) for t in tokens):
        return "", False
    if all(t[0].isupper() for t in tokens):
        return cleaned, False
    if allow_lower and len(tokens) <= 3 and not any(re.search(r"(?:ing|tion|ment|ly)$", t.lower()) for t in tokens):
        return cleaned, True
    return "", False


def _polarity(low: str) -> str:
    """yes, no or none. Mixed signals ("yes... actually no", "no, send it") are none."""
    if YES_THEN_NO_RE.match(low):
        return "no"  # "absolutely not", "ok no", "yeah no"
    if YES_START_RE.match(low):
        return "none" if NEGATION_RE.search(YES_START_RE.sub("", low, count=1)) else "yes"
    if NO_START_RE.match(low):
        return "none" if NO_BUT_RE.search(NO_START_RE.sub("", low, count=1)) else "no"
    return "none"


def _is_closing(low: str) -> bool:
    """A closing phrase with nothing but pleasantries around it ("no, that's all, thanks")."""
    if "?" in low or not DONE_RE.search(low):
        return False
    rest = re.findall(r"[a-z']+", DONE_RE.sub(" ", low))
    return all(word in _CLOSING_FILLER for word in rest)


def _email_choice(low: str, polarity: str, at_offer: bool, question_like: bool, leaving: bool = False) -> str:
    """send, skip or none. Mixed or unclear wording is none, so the policy asks again."""
    if re.search(r"\blater\b|\banother\s+time\b|\btomorrow\b|\bnext\s+week\b|\bnot\s+yet\b", low):
        return "none"  # nothing can be scheduled, so the question is asked again
    text = NOT_SKIP_RE.sub(" send it ", low)  # "don't skip it" asks for it
    skip = bool(EMAIL_NEGATED_RE.search(text))
    rest = EMAIL_NEGATED_RE.sub(" ", text)
    send = bool(EMAIL_SEND_RE.search(rest) or EMAIL_WISH_RE.search(rest))
    if skip and send:
        return "none"
    if skip:
        return "skip"
    if send:
        unclear = bool(NEGATION_RE.search(rest)) or (question_like and not EMAIL_REQUEST_RE.search(low))
        return "none" if unclear else "send"
    if not at_offer:
        return "none"
    if leaving or polarity == "no":
        return "skip"  # "ok bye" is someone leaving, not someone asking for an email
    if polarity == "yes":
        # A yes sends only when nothing else is said with it: "ok", "yes please", "sure, thanks".
        said = re.findall(r"[a-z'\-]+", YES_START_RE.sub("", low, count=1))
        return "send" if all(word in _YES_FILLER for word in said) else "none"
    return "none"


def states_third_party(text: str) -> bool:
    """The caller says outright that they act for somebody else ("I'm her husband", "calling for Jane Doe")."""
    if REP_STRONG_RE.search(text) or NOT_ME_RE.search(text):
        return True
    match = _POSSESSIVE_ROLE.search(text)
    if match and _name_tail(match.group(1) or match.group(2)):
        return True
    return any((m := pattern.search(text)) and _name_tail(m.group(1)) for pattern in (_CALLING_FOR, _CALLING_ABOUT, _ROLE_FOR, _IS_MY))


# --------------------------------------------------------------------------- the reading


def rule_understand(session: Session, text: str, data: ClaimsData, today: date) -> Understanding:
    u = Understanding(source="rules")
    low = text.lower().strip()
    memory = session.memory
    phase, pending = session.phase, session.pending
    in_verify = phase == VERIFY_ID
    words = low.split()
    letters = [ch for ch in text if ch.isalpha()]
    shouting = len(letters) > 12 and sum(ch.isupper() for ch in letters) / len(letters) > 0.7
    question_like = "?" in text or bool(QUESTION_START_RE.match(low))

    # ---- who is calling -----------------------------------------------------
    holder_name, own_after_role = "", ""
    match = _POSSESSIVE_ROLE.search(text)
    if match:
        holder_name = _name_tail(match.group(1) or match.group(2))
        if holder_name:
            own_after_role = _take_name(text[match.end():], True)  # "...'s son David Chen"
    third_party = ""
    for pattern in (_CALLING_FOR, _CALLING_ABOUT, _ROLE_FOR, _IS_MY):
        match = pattern.search(text)
        if match and _name_tail(match.group(1)):
            third_party = _name_tail(match.group(1))
            break
    if session.verified:
        if SELF_REP_RE.search(text):
            u.caller_role = "representative"
    elif REP_STRONG_RE.search(text) or holder_name or third_party:
        u.caller_role = "representative"
    elif POLICYHOLDER_RE.search(text):
        u.caller_role = "policyholder"
    elif REP_WEAK_RE.search(text) and (memory.caller_role == "representative" or not session.verification.get("matched")):
        u.caller_role = "representative"
        u.guessed.append("caller_role")  # inferred from how they talk, never said outright
    if not session.verified and u.caller_role == "unknown":
        if pending == "role_check":  # the agent has just asked "are you the policyholder yourself?"
            polarity = _polarity(low)
            says_no = polarity == "no" or bool(NOT_ME_RE.search(text))
            says_yes = polarity == "yes" or bool(ITS_ME_RE.search(text))
            if says_no and not says_yes:
                u.caller_role = "representative"
            elif says_yes and not says_no and not NEGATION_RE.search(low):
                u.caller_role = "policyholder"
        elif MENTIONS_OTHER_RE.search(text):
            u.mentions_other = True
    is_rep = u.caller_role == "representative" or (u.caller_role == "unknown" and memory.caller_role == "representative")

    # ---- structured identity values, masking each as it is found -------------
    raw: dict[str, str] = {}
    work = text
    hints: dict[str, object] = {}
    free_dates: list[tuple[int, int]] = []

    match = EMAIL_RE.search(work)
    if match:
        raw["email"] = match.group(0)
        work = _mask(work, match.span())
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
        someone_elses = bool(_THIRD_PARTY_RE.search(work[max(0, span[0] - 45): span[0]])) and not is_rep
        if someone_elses:
            pass  # "my husband's birthday is ...": not the caller's own detail
        elif DOB_TRIGGER_RE.search(before) or year <= today.year - 14:
            raw.setdefault("dob", f"{year:04d}-{month:02d}-{day:02d}")
        else:
            free_dates.append((year, month))
        work = _mask(work, span)
    match = PHONE_RE.search(work)
    if match and 10 <= len(re.sub(r"\D", "", match.group(1))) <= 11:
        before = work[max(0, match.start() - 45): match.start()]
        if not _CALLBACK_RE.search(before) and not (_THIRD_PARTY_RE.search(before) and not is_rep):
            raw["phone"] = match.group(1)
        work = _mask(work, match.span())
    match = POLICY_RE.search(work)
    if match:
        raw["policy_number"] = match.group(1)
        work = _mask(work, match.span())
    if "id_last4" not in raw:
        match = LAST4_RE.search(work)
        if match:
            raw["id_last4"] = match.group(1)
            work = _mask(work, match.span(1))
    structured = bool(raw)  # something that is unmistakably an identity value

    # ---- names ----------------------------------------------------------------
    self_name = _first_name(_STRONG_NAME, text, False)
    lower_guess = False
    if not self_name and not shouting:
        weak = _first_name(_WEAK_NAME, text, True)
        first_only = bool(weak) and in_verify and pending == "identity" and weak.lower() not in _COMMON
        self_name = weak if len(weak.split()) >= 2 or first_only else ""
        if not self_name and in_verify and pending in {"identity", "rep_name"}:
            # "hi this is jane doe": typed without capitals, so it is only a guess
            weak = _first_name(_WEAK_NAME, text, False)
            tokens = weak.split()
            plain = all(t.lower() not in _COMMON and not re.search(r"(?:ing|ed|ly)$", t.lower()) for t in tokens)
            if 2 <= len(tokens) <= 3 and plain:
                self_name, lower_guess = weak, True
    if not self_name:
        match = _SUFFIX_NAME.search(text)
        if match and not any(tok.lower() in _COMMON for tok in match.group(1).split()):
            self_name = match.group(1)
    if holder_name and self_name.lower() == holder_name.lower():
        self_name = ""  # "I'm Jane Doe's husband" names Jane Doe, not the caller
    if len(own_after_role.split()) >= 2:
        self_name = own_after_role
    other_name = holder_name or third_party or _first_name(_OTHER_NAME, text, True)
    known_holder = (memory.identity.get("full_name") or "").lower()
    if is_rep:
        if other_name:
            raw["full_name"] = other_name
        if self_name and self_name.lower() == known_holder:
            u.guessed.append("self_is_holder")  # they give the policyholder's name as their own
        elif len(self_name.split()) >= 2 and self_name.lower() != other_name.lower():
            u.representative_name = self_name
    elif self_name:
        raw["full_name"] = self_name
        if lower_guess:
            u.guessed.append("full_name")

    # A reply that is only a name, possibly next to other details: "Jane Doe, 0000"
    if in_verify:
        dob_known = "dob" in raw or "dob" in memory.identity
        known_name = (memory.identity.get("full_name") or "").lower()
        segments = [seg.strip() for seg in re.split(r"[,;\n]|\band\b|[.!?](?:\s|$)", work) if seg.strip()]
        for piece in segments:
            # a four-digit number standing alone, possibly after a name: "jane doe 0000"
            match = re.fullmatch(r"\W*(?:it'?s\s+|its\s+)?(?:([A-Za-z][A-Za-z'\-\s]*?)\s+)?(\d{4})\W*", piece, _I)
            if match and (match.group(1) or "").strip():
                lower_fine = pending in {"identity", "rep_name"} and not question_like
                if not _bare_name(match.group(1).strip(), allow_lower=lower_fine)[0]:
                    match = None  # "my claim from 2025", "my phone ends in 2836": not a name, so not an ID
            if match and "id_last4" not in raw:
                number = int(match.group(2))
                if 1900 <= number <= today.year and not dob_known:
                    # Could be a year of birth. Never guess: a wrong guess is a failed attempt.
                    u.partial.append("dob" if re.search(r"\bborn\b|\bbirth", low) else "dob_or_id")
                else:
                    raw["id_last4"] = match.group(2)
                    structured = True
                piece = (match.group(1) or "").strip()
                if not piece:
                    continue
            lower_ok = pending in {"identity", "rep_name"} and (len(segments) == 1 or structured) and not question_like
            name, guess = _bare_name(piece, allow_lower=lower_ok)
            if not name and pending == "identity" and re.fullmatch(r"\W*[A-Z][a-z\-]{1,20}\W*", piece):
                word = piece.strip(" .,;:!?")
                if word.lower() not in _COMMON and word.lower() not in MONTHS:
                    name, guess = word, True  # one capitalised word: part of a name
            if not name:
                continue
            if pending == "role_check" and known_name and name.lower() != known_name and not guess:
                # asked "are you the policyholder yourself?", they answer with another name
                u.representative_name = u.representative_name or name
                u.caller_role = "representative"
            elif pending == "rep_name":
                if name.lower() == known_name:
                    u.guessed.append("self_is_holder")
                else:
                    u.representative_name = u.representative_name or name
            elif "full_name" not in raw and not (is_rep and u.representative_name == name):
                raw["full_name"] = name
                if guess:
                    u.guessed.append("full_name")
        if pending == "identity" and not dob_known:
            # a date without its year, or digits that are not a whole date or phone number
            partial_date = any(year is None for year, _m, _d, _s in find_dates(work)) and not re.search(r"\bclaims?\b", low)
            if partial_date or re.fullmatch(r"\W*\d{1,2}\s*[/.\-]\s*\d{1,2}\W*", work.strip()):
                u.partial.append("dob")
            elif re.search(r"\bborn\s+in\b|\byear\s+of\s+birth\b", low) and "dob_or_id" not in u.partial:
                u.partial.append("dob")
        if pending == "identity" and "phone" not in raw and re.fullmatch(r"[\d\s().+\-]+", work.strip() or "x"):
            if 7 <= len(re.sub(r"\D", "", work)) <= 9:
                u.partial.append("phone")
        u.partial = list(dict.fromkeys(u.partial))

    u.identity = normalize_identity(raw, today)
    u.representative_name = clean_representative_name(u.representative_name)
    identity_given = any(k in u.identity for k in ("full_name", "dob", "phone", "email", "id_last4"))

    # ---- declined details, refusal, questions about the call ---------------------
    u.process_question = bool(PROCESS_Q_RE.search(low) or BOT_QUESTION_RE.search(low))
    if not session.verified:
        for field_name, pattern in DECLINE_FIELDS:
            if field_name in u.identity:
                continue
            if any(re.search(template.format(field=pattern), low) for template in DECLINE_TEMPLATES):
                u.declined_fields.append(field_name)
        if "full_name" in u.declined_fields and len(u.declined_fields) > 1:
            u.declined_fields.remove("full_name")  # "name" is too generic when another detail is named
    if not session.verified and not identity_given and not u.declined_fields:
        says_no = pending == "identity" and (PLAIN_NO_RE.match(low) or (_polarity(low) == "no" and not question_like))
        if REFUSAL_RE.search(low) or says_no:
            u.refuses_verification = True

    # ---- what the caller wants ---------------------------------------------------
    found = [(m.start(), intent) for intent, pattern in INTENT_RULES if (m := re.search(pattern, low))]
    matched = [intent for _, intent in sorted(found, key=lambda item: item[0])]
    if phase == PROCESS_CASE:
        if re.fullmatch(r"\W*(?:but\s+)?why(?:\s+not|\s+is\s+that|\s+though)?\W*", low):
            matched = ["denial_explanation"]
        elif DISPUTE_RE.search(low) and "appeal_or_next_steps" not in matched:
            matched.append("appeal_or_next_steps")
        if not matched and re.search(r"\bwhat\s+happens\s+(?:next|now|then|after)\b|\bwhat(?:'s|\s+is)\s+next\b", low):
            matched = ["appeal_or_next_steps"]
    if WHAT_NEED_RE.search(low):
        if in_verify:
            u.process_question = True
        elif "document_requirements" not in matched:
            matched.append("document_requirements")
    # "send it" at the email offer is about the email, not about documents
    summary_open = pending == "email_offer" or phase == CLOSED
    if "document_requirements" in matched and (summary_open or re.search(r"\be-?mails?\b|\bsummary\b|\brecap\b", low)):
        if not re.search(INTENT_RULES[2][1], EMAILISH_RE.sub(" ", low)):
            matched.remove("document_requirements")

    topics: list[str] = []
    for row in data.guidelines.get("claim_followup_guidance", []):
        if any(re.search(rf"\b{re.escape(phrase)}", low) for phrase in row.get("match_any") or []):
            topics.append(row["topic"])
    if re.search(r"\b(?:how|where)\s+(?:do|can|should|would)\s+i\s+(?:send|submit|upload)\b|\bfax\b|(?<![e\-])\bmail\s+(?:it|them)\b", low):
        topics.append("submission_method")
    if re.search(
        r"how\s+(?:will|do|would)\s+i\s+know\b|did\s+you\s+(?:get|receive)\b|confirm\w*\b.{0,30}\b(?:receiv|got|get)|"
        r"\bi(?:'ve|\s+have)?\s+(?:already\s+|just\s+)?(?:mailed|sent|uploaded|submitted|faxed)\b",
        low,
    ):
        topics.append("receipt_confirmation")
    about_documents = not matched or matched[0] in {"document_requirements", "appeal_or_next_steps"}
    if ALTERNATIVE_RE.search(low) and about_documents and not u.declined_fields:
        if not in_verify or re.search(r"report|note|document|estimate|photo", low):
            topics.append("missing_required_material_alternatives")
    u.followup_topics = list(dict.fromkeys(topics))
    if u.followup_topics and not matched and phase in {PROCESS_CASE, RESOLVE_INTENT}:
        matched = ["document_requirements"]
    action_request = session.verified and bool(ACTION_REQUEST_RE.search(low)) and not NEGATION_RE.search(low)
    if action_request:
        matched, u.followup_topics = [], []  # no path does this: the reply says so and offers a person
    elif not matched and GENERAL_CLAIM_RE.search(low) and not u.process_question:
        matched = ["general_claim_question"]

    # ---- which claim -----------------------------------------------------------------
    hint_text = work.lower()
    if session.case_id and session.verified:
        # A claim is in focus. Only a hint attached to a claim reference can point elsewhere:
        # "my auto claim", "the one from 2025". A stray date or the word "car" cannot.
        dismissed = re.search(
            r"\bnot\s+(?:about|for|on|regarding)\b|n't\s+about\b|\bbut\s+that's\s+(?:fine|ok|okay)\b|\bnever\s?mind\b|"
            r"\bnot\s+now\b|\bforget\s+(?:it|that|about)\b",
            hint_text,
        )
        if not dismissed and not re.search(r"\bthis\s+(?:claim|one|case)\b", hint_text):
            for claim_type, alt in _TYPE_WORDS.items():
                if re.search(rf"\b(?:{alt})\s+(?:\w+\s+)?{_REF}\b", hint_text):
                    hints["claim_type"] = claim_type
                    break
            match = STATUS_REF_RE.search(hint_text)
            if match:
                hints["status"] = next(s for s, pattern in STATUS_RULES if re.search(pattern, match.group(1)))
            match = MONTH_REF_RE.search(hint_text)
            if match:
                hints["month"] = MONTHS[(match.group(1) or match.group(3)).lower()]
                year_text = match.group(2) or match.group(4)
                if year_text:
                    hints["year"] = int(year_text)
            match = YEAR_REF_RE.search(hint_text)
            if match:
                value = (match.group(1) or match.group(2)).replace("'s", "")
                hints["year"] = today.year - 1 if "last" in value else today.year if "this" in value else int(value)
            if FOLLOW_ON_RE.search(hint_text) and "claim_id" not in hints:
                hints["exclude_current"] = True
            if "status" in hints and len(hints) == 1:
                # "the denied one" while an open claim is in focus points at another claim
                current = data.claim_for_party(session.party_id, session.case_id)
                if current and current["status"] != hints["status"]:
                    hints["exclude_current"] = True
            if "status" not in hints and any(k in hints for k in ("claim_type", "month", "year", "exclude_current")):
                # The message is about another claim, so a status word in it describes that one.
                for status, pattern in STATUS_RULES:
                    if re.search(pattern, hint_text):
                        hints["status"] = status
                        break
    else:
        claim_context = (
            bool(re.search(r"\bclaims?\b|\bcase\b|\bdenied\b|\bdenial\b|\brejected\b|\bappeal", low))
            or pending == "choose_case"
            or phase == RESOLVE_INTENT
            or "claim_id" in hints
        )
        if claim_context:
            for claim_type, alt in _TYPE_WORDS.items():
                if re.search(rf"\b(?:{alt})\b", hint_text):
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
            if free_dates:
                hints.setdefault("month", free_dates[0][1])
                hints.setdefault("year", free_dates[0][0])
            if pending == "choose_case" and "claim_id" not in hints:
                # an answer to "which one?": "the november one", "February 28", "the 2026 one"
                for _year, month, _day, _span in find_dates(hint_text):
                    hints.setdefault("month", month)
                match = BARE_MONTH_RE.search(hint_text)
                if match and (match.group(1) != "may" or re.search(r"\bmay\s+(?:one|claim|\d)|\bin\s+may\b", hint_text)):
                    hints.setdefault("month", MONTHS[match.group(1)])
                match = BARE_YEAR_RE.search(hint_text)
                if match:
                    hints.setdefault("year", int(match.group(1)))
    u.case_hints = hints

    if pending == "choose_case" and session.last_candidates and "claim_id" not in hints:
        match = ORDINAL_RE.search(low) or re.fullmatch(r"\W*(?:number\s*|#)?([1-9])\W*", low)
        if match:
            key = re.sub(r"\s+", " ", next(g for g in match.groups() if g).lower())
            index = _ORDINALS.get(key, int(key) - 1 if key.isdigit() else None)
            if index is not None and -1 <= index < len(session.last_candidates):
                u.selected_case_id = session.last_candidates[index]

    # ---- feelings ------------------------------------------------------------------------
    for emotion, pattern in EMOTION_RULES:
        if re.search(pattern, low):
            u.emotion = emotion
            break
    if u.emotion == "neutral" and shouting:
        u.emotion = "angry"

    # ---- a person, yes or no, more to ask, finished ---------------------------------------
    clauses = [c for c in re.split(r"[.!?;,]+|\bbut\b|\bso\b", text) if c.strip()]
    u.wants_human = (
        any(
            HUMAN_RE.search(clause) and not HUMAN_NEG_RE.search(clause) and not HUMAN_ASKING_RE.search(clause)
            for clause in clauses
        )
        and not REP_STATEMENT_RE.search(low)
        and not BOT_QUESTION_RE.search(low)
    )
    u.affirmation = _polarity(low)
    if pending == "human_offer" and u.affirmation == "yes":
        said = re.findall(r"[a-z'\-]+", YES_START_RE.sub("", low, count=1))
        if not all(word in _YES_FILLER or word in _HANDOFF_WORDS for word in said):
            u.affirmation = "none"  # "sure, thanks anyway", "alright, I'll call back later"
    asks = bool(matched or u.followup_topics)
    u.more = bool(MORE_RE.search(low)) and not asks
    if u.more:
        u.affirmation = "none"
    elif not asks:
        if _is_closing(low):
            u.done = True
        elif pending == "anything_else" and (PURE_NO_RE.match(low) or THANKS_ONLY_RE.match(low)):
            u.done = True
    if u.done:
        u.refuses_verification = False  # "forget it, goodbye" is leaving, not arguing

    # ---- the email summary ------------------------------------------------------------------
    at_offer = pending == "email_offer"
    # At the offer, a question that is not about a claim is a question about the summary.
    summary_question = at_offer and question_like and not asks and not u.case_hints and not EMAIL_REQUEST_RE.search(low)
    if at_offer and (ADDRESS_CHANGE_RE.search(low) or u.process_question or summary_question):
        u.process_question = True  # a question about the summary, not a choice
        u.affirmation = "none"  # "yes, but to my work address" is not a yes to sending
        matched = [i for i in matched if re.search(r"\bdocuments?\b|\bclaims?\b", low)]
    elif at_offer or phase == CLOSED:
        # After closing, only wording that names the summary reopens the choice. A preference
        # is never read there, because in CLOSED a "send" is an action.
        reopen = phase == CLOSED and not session.outbox and bool(CLOSED_SEND_RE.search(low))
        if (at_offer or reopen) and not asks and not u.more:
            u.email_summary_choice = _email_choice(low, u.affirmation, at_offer, question_like, leaving=u.done)
            if at_offer and u.email_summary_choice == "none":
                u.affirmation = "none"  # the yes or no is already weighed above: "yes, later" is not a yes
    elif EMAIL_VOLUNTEER_RE.search(low) and "email" not in u.identity and not (in_verify and u.declined_fields):
        # Said at another point in the call: a preference to remember, never an action.
        if EMAIL_NEGATED_RE.search(low):
            u.email_summary_choice = "skip"
        elif re.search(r"\b(?:send|e-?mail|mail|get|have|want|like)\b", low) and not u.process_question:
            if not NEGATION_RE.search(low):
                u.email_summary_choice = "send"
    if u.email_summary_choice != "none" and not re.search(r"\bdocuments?\b|\bclaims?\b", low):
        matched = [i for i in matched if i not in {"document_requirements", "general_claim_question"}]

    if matched:
        u.intent = matched[0]
        u.extra_intents = [i for i in matched[1:] if i != u.intent][:2]
        u.question = text.strip()[:300]
    elif u.followup_topics:
        u.question = text.strip()[:300]

    # ---- scope --------------------------------------------------------------------------------
    short_answer = u.affirmation != "none" or bool(GREETING_RE.match(low)) or len(words) <= 1
    domain_hit = bool(
        u.identity or u.representative_name or u.intent != "none" or u.case_hints or u.selected_case_id
        or u.process_question or u.wants_human or u.email_summary_choice != "none" or u.done or u.more
        or u.refuses_verification or u.declined_fields or u.followup_topics or u.caller_role != "unknown"
        or u.partial or DOMAIN_RE.search(low)
    )
    oos_marker = bool(OOS_RE.search(low)) or bool(MATH_RE.match(work.strip()))
    general = bool(GENERAL_KNOWLEDGE_RE.match(low) or GENERAL_TASK_RE.match(low))
    unknown_question = question_like and not domain_hit and not short_answer
    claimish = bool(CLAIMISH_RE.search(low))
    # Something ties the message to a claim or to what was just said.
    related = claimish or bool(ANAPHORA_RE.search(low) or CLARIFY_RE.search(low)) or len(words) <= 2
    about_step = False
    if not session.verified and not identity_given and u.intent == "none" and not u.wants_human and not u.done:
        # Before verification a question about the step at hand ("which three?", "can I use
        # my address instead?") is answered, never declined.
        clarifies = u.affirmation == "none" and bool(SHORT_CLARIFY_RE.match(low))
        about_step = bool(
            (question_like and VERIFY_Q_RE.search(low)) or clarifies or WAIT_RE.search(low)
            or (unknown_question and pending in {"consent", "rep_name"})
        )
    if oos_marker and not domain_hit:
        u.scope = "out_of_scope"
    elif general and not domain_hit and not claimish:
        u.scope = "out_of_scope"  # "what is the capital of ...", not "what was the date of service?"
    elif oos_marker:
        u.scope = "mixed"
    elif unknown_question and not related and not about_step:
        u.scope = "out_of_scope"  # nothing ties it to a claim, to the step at hand or to the conversation
    if u.scope == "out_of_scope":
        u.affirmation = "none"
        return u

    if not session.verified:
        if not identity_given and u.intent == "none" and not u.wants_human and not u.done:
            if about_step:
                u.process_question = True
            elif unknown_question and claimish:
                u.question = text.strip()[:300]  # something to help with once verified
    elif action_request and not u.question:
        u.question = text.strip()[:300]
    elif question_like and not short_answer and not u.question and not u.process_question:
        if u.emotion == "neutral" and CLARIFY_RE.search(low) and not claimish:
            u.emotion = "confused"  # "are you sure?", "how so?": go over it again, do not hand off
        about_email = at_offer and bool(EMAIL_WORDS_RE.search(low))
        picks_claim = bool(u.selected_case_id) or (pending == "choose_case" and bool(u.case_hints))
        settled = u.email_summary_choice != "none" or u.more or u.done or u.wants_human or about_email or picks_claim
        if u.emotion != "confused" and not settled:
            u.question = text.strip()[:300]  # in scope, but no rule knows how to answer it
    return u
