# HinEmo-9 — Annotation Guidelines

**Dataset:** English–Hindi–Hinglish YouTube comment emotion dataset
**Round:** Main annotation (3 annotators: A01, A02, A03)
**Version:** 1.0

---

## 1. What you are doing

You will read short YouTube comments (English, Hindi in Devanagari, or romanized Hinglish) and label the emotion the **comment writer is expressing**.

Label what the writer *feels or expresses*, not:

- what the video was about,
- what emotion the writer is *describing* in someone else,
- what you would feel reading it.

> "Is video mein log bahut dar gaye the" — the writer is *reporting* that people were scared; the writer is not expressing fear. This is **Neutral**.
> "Bhai main to dar gaya 😨" — the writer *is* expressing fear. This is **Fear**.

---

## 2. What you fill in for each comment

| Field | What to enter |
|---|---|
| **primary_emotion** | Exactly one label — the dominant emotion. Required. |
| **other_emotions** | Any additional emotions clearly present. Zero, one, or more. Optional. |
| **emotion_carrying_language** | Which language carries the emotional content: `en` / `hi` / `both` / `neither` |
| **confidence** | How sure you are: High / Medium / Low |
| **notes** | Free text — use for anything odd, ambiguous, or worth flagging |

**Every comment gets exactly one primary_emotion.** If nothing emotional is present, that label is **Neutral** — Neutral is a real choice, not a fallback for "I'm not sure."

---

## 3. The nine labels

### Love
Affection, attachment, romantic feeling, warmth toward a person. Directed at an equal or a peer.
> "Tumhare bina din adhura lagta hai ❤️"
> "मेरी जान, तुम्हारा हर गाना दिल छू जाता है"

### Joy
Happiness, delight, amusement, celebration, enjoyment. Includes laughter and appreciation of entertainment.
> "आपकी वीडियो बहुत अच्छी है 🥰"
> "Bhai maza aa gaya 😂😂"

### Anger
Irritation, rage, indignation, disgust, blame, hostility, betrayal-anger.
> "Itna ghatiya content banate ho, sharam nahi aati"
> "धोखा देने वालों को कभी माफ़ नहीं करना चाहिए"

### Sadness
Grief, hurt, loneliness, disappointment, heartbreak — **about the present or an ongoing state**.
> "Bahut rulaya yaar 😢😢"
> "अकेलापन सबसे बड़ा दर्द है"

### Fear
Fright, dread, anxiety, worry, feeling scared or unsafe.
> "Dara dia bhai 😭😭"
> "Main bhi nurse hu maine hospital me spirit ko dekha, ab bhi darr lagta hai"

### Surprise
Shock, astonishment, disbelief, being caught off-guard. Can be positive or negative.
> "Bedroom mein cctv koun rakhta hai...?! 😲"
> "यकीन नहीं हो रहा ये सच है"

### Nostalgia *(novel label)*
**Bittersweet longing for the past.** Missing something gone, fond remembering with an ache.
> "Bade din huye aisa gaana suna nahi 🥹"
> "बचपन के वो दिन फिर कभी नहीं आएंगे"

### Devotion *(novel label)*
**Reverence toward something higher** — deity, guru, nation, a revered figure. Asymmetric: the writer looks *up* at the object.
> "जय श्री राम 🙏🙏"
> "सदगुरु को मेरा प्रणाम"

### Neutral
No emotional expression. Facts, questions, "who's watching from X", timestamps, requests, plain commentary.
> "राजस्थान से कौन कौन देख रहा है"
> "Part 2 kab aayega?"

---

## 4. The three boundaries that matter most

These are where annotators most often disagree. Read these carefully.

### Nostalgia vs. Sadness
The test is **time orientation plus warmth**.

- **Nostalgia** = looking *back* at something good that is gone, with fondness mixed into the ache.
- **Sadness** = present pain, no fond backward warmth required.

| Comment | Label | Why |
|---|---|---|
| "पुराने दिन याद आ गए, कितने अच्छे थे" | Nostalgia | past + fondness |
| "आज बहुत बुरा दिन था, मन उदास है" | Sadness | present, no fondness |
| "Papa ko yaad karke aankh bhar aayi, kitna hasate the" | Nostalgia | past + warmth (grief with fondness) |
| "Papa nahi rahe, kuch acha nahi lagta" | Sadness | present loss, no fond recall |

If a comment has past-fondness **and** present pain, choose **Nostalgia** as primary and add **Sadness** to other_emotions.

### Devotion vs. Love
The test is **whether the relationship is equal or looking-upward**.

- **Devotion** = reverence, worship, salutation, submission. Deity, guru, nation, martyrs, deeply revered figure.
- **Love** = affection between equals — partner, friend, family, a creator you're fond of.

| Comment | Label | Why |
|---|---|---|
| "हर हर महादेव 🙏" | Devotion | worship |
| "Bhai tumhara content dil se pasand hai ❤️" | Love | peer affection |
| "माँ तुम्हारे चरणों में प्रणाम" | Devotion | reverential framing |
| "Maa I love you, tum best ho" | Love | affectionate, not reverential |

Praise of a creator ("bahut sundar gaya aapne") is usually **Joy** or **Love**, not Devotion — *unless* the framing is reverential (प्रणाम, नमन, चरण वंदना).

### Fear vs. Surprise
Both are reactions to something sudden. The test is **threat**.

- **Fear** = the writer feels *unsafe or threatened*.
- **Surprise** = the writer is *astonished*, without feeling endangered.

| Comment | Label |
|---|---|
| "Achanak aisa hua, main sehem gaya 😨" | Fear |
| "Achanak aisa hua, yakeen hi nahi hua 😲" | Surprise |

Devotional comments on scary content (e.g. "जय श्री राम 🙏" under a horror video) are **Devotion**, not Fear. Label the comment, not the video.

---

## 5. Primary vs. other emotions

- **primary_emotion** = the single strongest emotion. If two feel equal, pick the one the comment *ends on* or emphasizes most.
- **other_emotions** = clearly present but secondary. Do not add an emotion just because it's faintly imaginable — only if a reader would recognize it in the text.

> "Purane gaane sunke rona aa gaya, but kya din the yaar 🥹"
> primary = **Nostalgia**, other = **Sadness**

---

## 6. emotion_carrying_language

Which language actually carries the emotional content?

| Value | Use when |
|---|---|
| `en` | The emotional words are English — "this is so scary" |
| `hi` | The emotional words are Hindi (Devanagari or romanized) — "बहुत डर लगा", "bahut dar laga" |
| `both` | Emotional content in both — "bahut scary tha yaar, so creepy" |
| `neither` | No emotional language (usually Neutral comments, or emotion carried only by emoji) |

Note: **romanized Hindi counts as Hindi (`hi`)** — "dar gaya" is Hindi, just typed in Latin script.

---

## 7. Edge cases

| Situation | What to do |
|---|---|
| **Emoji only** ("😂😂😂") | Label from the emoji (here: Joy). ECL = `neither` |
| **`[NAME]` masks** | Names are anonymized. Ignore the mask, label the rest |
| **Sarcasm** ("Wah, kya bakwaas video 👏") | Label the *intended* emotion — here Anger, not Joy. Flag in notes |
| **Mixed/multiple emotions** | Strongest → primary, rest → other_emotions |
| **Not Hindi or English** (Bhojpuri, Punjabi, Bengali, etc.) | Label emotion normally; note the language in notes |
| **Spam / promotion / "subscribe my channel"** | Neutral |
| **Truncated or unintelligible** | Neutral, confidence = Low, flag in notes |
| **Genuinely can't decide** | Pick your best guess, set confidence = Low, explain in notes |

Do **not** skip comments. Every task gets a label.

---

## 8. About the pre-filled predictions ⚠️

Each comment arrives with a **machine-generated suggestion** already selected. This exists to save you time — it is **not** the answer.

**Please:**

- Read the comment **first**, form your own judgement, *then* look at the suggestion.
- Override it whenever you disagree. It is wrong a meaningful share of the time.
- Be especially skeptical on **Nostalgia, Devotion, Fear, and Surprise** — the model is weakest on exactly these.

If you find yourself accepting nearly every suggestion, stop and re-calibrate. Agreement that comes from following the machine rather than reading the text weakens the dataset.

---

## 9. Practical notes

- Work in **your own project only**. Do not discuss specific comments with the other annotators during the round — independence is what makes the agreement statistics meaningful.
- Aim for steady sessions (roughly 200–400 comments) rather than long marathons; fatigue degrades label quality.
- Keep a running list of recurring hard cases and raise them at the scheduled calibration checkpoints — guideline updates happen there, not mid-flow.
- When guidelines are updated, the updated version applies **going forward**; earlier annotations are revisited only if the review decides so.

---

## Quick reference

**Nostalgia** = past + fondness · **Sadness** = present pain
**Devotion** = looking up (reverence) · **Love** = between equals
**Fear** = feels threatened · **Surprise** = astonished, not threatened
**Neutral** = a genuine label, not a "don't know"
**Romanized Hindi** = `hi`, not `en`
**Label the writer's emotion**, not the video's topic
