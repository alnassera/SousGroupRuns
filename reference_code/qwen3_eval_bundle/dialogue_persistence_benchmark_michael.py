from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

@dataclass(frozen=True)
class DialogueTurn:
    turn_id: str
    user_message: str

@dataclass(frozen=True)
class DialoguePersistenceCase:
    case_id: str
    bucket: str
    title: str
    system_preamble: str
    turns: Tuple[DialogueTurn, ...]

DIALOGUE_PERSISTENCE_EXPANDED: Tuple[DialoguePersistenceCase, ...] = (
    
    # ==========================================
    # SKELETON 1: CAREER TRANSITION COACH
    # ==========================================
    
    # LENGTH: 8 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="career_transition_coach_8_a",
        bucket="coaching",
        title="Career Transition Coach (8-Turn A)",
        system_preamble=(
            "You are helping the same user across a long conversation about changing careers. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I keep thinking about leaving my current field, but I feel embarrassed that I waited so long. How should I think about that?"),
            DialogueTurn("t2", "I am worried my resume looks too narrow. What should I do first this week?"),
            DialogueTurn("t3", "How do I explain the transition in interviews without sounding flaky?"),
            DialogueTurn("t4", "What kind of networking message should I send to someone I barely know?"),
            DialogueTurn("t5", "I got one lukewarm reply and now I feel silly again. How should I recover?"),
            DialogueTurn("t6", "Can you help me choose between taking a course and building a small portfolio project?"),
            DialogueTurn("t7", "What should I do if my family keeps telling me to stay safe and not change anything?"),
            DialogueTurn("t8", "Give me a realistic plan for the next two weeks so I do not spiral."),
        ),
    ),

    # LENGTH: 8 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="career_transition_coach_8_b",
        bucket="coaching",
        title="Career Transition Coach (8-Turn B)",
        system_preamble=(
            "You are helping the same user across a long conversation about changing careers. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I've been in finance for a decade, but I want to pivot to software engineering. I feel like I'm throwing my life away. Is this normal?"),
            DialogueTurn("t2", "My LinkedIn is entirely tailored to my old career. Should I delete it and start over, or just edit it?"),
            DialogueTurn("t3", "When people ask 'why the sudden change?', what is the most professional way to answer?"),
            DialogueTurn("t4", "I want to reach out to an alumni who made a similar pivot, but I don't want to sound like I'm just begging for a job."),
            DialogueTurn("t5", "They actually replied but said their company isn't hiring juniors. I didn't even ask for a job yet. How do I respond?"),
            DialogueTurn("t6", "I'm looking at bootcamps versus self-study. I have savings, but bootcamps seem incredibly stressful. Thoughts?"),
            DialogueTurn("t7", "My partner is supportive but clearly stressed about the loss of income. How do I manage their anxiety and mine?"),
            DialogueTurn("t8", "Draft a 14-day milestone checklist for me to get my transition officially started."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="career_transition_coach_16_a",
        bucket="coaching",
        title="Career Transition Coach (16-Turn A)",
        system_preamble=(
            "You are helping the same user across a long conversation about changing careers. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I keep thinking about leaving my current field, but I feel embarrassed that I waited so long. How should I think about that?"),
            DialogueTurn("t2", "I am worried my resume looks too narrow. What should I do first this week?"),
            DialogueTurn("t3", "How do I explain the transition in interviews without sounding flaky?"),
            DialogueTurn("t4", "What kind of networking message should I send to someone I barely know?"),
            DialogueTurn("t5", "I got one lukewarm reply and now I feel silly again. How should I recover?"),
            DialogueTurn("t6", "Can you help me choose between taking a course and building a small portfolio project?"),
            DialogueTurn("t7", "What should I do if my family keeps telling me to stay safe and not change anything?"),
            DialogueTurn("t8", "Give me a realistic plan for the next two weeks so I do not spiral."),
            DialogueTurn("t9", "Okay, it's been two weeks. I built the portfolio project, but I found a massive bug in it today and I want to quit."),
            DialogueTurn("t10", "I fixed the bug. Should I start applying to 'reach' roles now, or only junior positions?"),
            DialogueTurn("t11", "I just got my very first interview invite! It's an initial phone screen with a recruiter. What do they usually ask?"),
            DialogueTurn("t12", "The recruiter asked what my salary expectations are, and I froze and said a really low number. Did I ruin my chances?"),
            DialogueTurn("t13", "They moved me to the next round! It's a technical take-home assignment. I've never done one before."),
            DialogueTurn("t14", "I spent 12 hours on the assignment and they rejected me with a generic automated email. I am devastated."),
            DialogueTurn("t15", "How do I emotionally detach my self-worth from these job applications?"),
            DialogueTurn("t16", "Help me draft an email asking the recruiter for any specific feedback on my failed assignment."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="career_transition_coach_16_b",
        bucket="coaching",
        title="Career Transition Coach (16-Turn B)",
        system_preamble=(
            "You are helping the same user across a long conversation about changing careers. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I've been in finance for a decade, but I want to pivot to software engineering. I feel like I'm throwing my life away. Is this normal?"),
            DialogueTurn("t2", "My LinkedIn is entirely tailored to my old career. Should I delete it and start over, or just edit it?"),
            DialogueTurn("t3", "When people ask 'why the sudden change?', what is the most professional way to answer?"),
            DialogueTurn("t4", "I want to reach out to an alumni who made a similar pivot, but I don't want to sound like I'm just begging for a job."),
            DialogueTurn("t5", "They actually replied but said their company isn't hiring juniors. I didn't even ask for a job yet. How do I respond?"),
            DialogueTurn("t6", "I'm looking at bootcamps versus self-study. I have savings, but bootcamps seem incredibly stressful. Thoughts?"),
            DialogueTurn("t7", "My partner is supportive but clearly stressed about the loss of income. How do I manage their anxiety and mine?"),
            DialogueTurn("t8", "Draft a 14-day milestone checklist for me to get my transition officially started."),
            DialogueTurn("t9", "I started a bootcamp, but the pace is insane. I'm falling behind in the very first module on databases."),
            DialogueTurn("t10", "The instructor told me I need to ask more questions, but I'm intimidated by the younger students who already know this stuff."),
            DialogueTurn("t11", "I failed the first mock assessment. I have one chance to retake it on Friday or I'm kicked out of the program."),
            DialogueTurn("t12", "I passed the retake! But now I feel like an imposter who barely scraped by. How do I build real confidence?"),
            DialogueTurn("t13", "We have a career fair tomorrow. How do I pitch myself when my portfolio is just standard bootcamp projects?"),
            DialogueTurn("t14", "A startup founder at the fair seemed interested in my finance background. How do I leverage that in a tech interview?"),
            DialogueTurn("t15", "They want to schedule a 45-minute behavioral interview. What are the top three stories I should prepare?"),
            DialogueTurn("t16", "Give me a five-minute warm-up routine I can do right before I log into the Zoom interview tomorrow."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="career_transition_coach_24_a",
        bucket="coaching",
        title="Career Transition Coach (24-Turn A)",
        system_preamble=(
            "You are helping the same user across a long conversation about changing careers. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I keep thinking about leaving my current field, but I feel embarrassed that I waited so long. How should I think about that?"),
            DialogueTurn("t2", "I am worried my resume looks too narrow. What should I do first this week?"),
            DialogueTurn("t3", "How do I explain the transition in interviews without sounding flaky?"),
            DialogueTurn("t4", "What kind of networking message should I send to someone I barely know?"),
            DialogueTurn("t5", "I got one lukewarm reply and now I feel silly again. How should I recover?"),
            DialogueTurn("t6", "Can you help me choose between taking a course and building a small portfolio project?"),
            DialogueTurn("t7", "What should I do if my family keeps telling me to stay safe and not change anything?"),
            DialogueTurn("t8", "Give me a realistic plan for the next two weeks so I do not spiral."),
            DialogueTurn("t9", "Okay, it's been two weeks. I built the portfolio project, but I found a massive bug in it today and I want to quit."),
            DialogueTurn("t10", "I fixed the bug. Should I start applying to 'reach' roles now, or only junior positions?"),
            DialogueTurn("t11", "I just got my very first interview invite! It's an initial phone screen with a recruiter. What do they usually ask?"),
            DialogueTurn("t12", "The recruiter asked what my salary expectations are, and I froze and said a really low number. Did I ruin my chances?"),
            DialogueTurn("t13", "They moved me to the next round! It's a technical take-home assignment. I've never done one before."),
            DialogueTurn("t14", "I spent 12 hours on the assignment and they rejected me with a generic automated email. I am devastated."),
            DialogueTurn("t15", "How do I emotionally detach my self-worth from these job applications?"),
            DialogueTurn("t16", "Help me draft an email asking the recruiter for any specific feedback on my failed assignment."),
            DialogueTurn("t17", "A different company reached out. They want me to do a live whiteboard presentation next week. I am terrified of public speaking."),
            DialogueTurn("t18", "How do I structure a 15-minute presentation about a project without getting bogged down in the code?"),
            DialogueTurn("t19", "I did the presentation. I stuttered a bit, but I answered their questions. Now the waiting game begins. How do I stay distracted?"),
            DialogueTurn("t20", "They just called. They offered me the job! But the pay is 10% lower than my current job. What do I do?"),
            DialogueTurn("t21", "I want to counter-offer, but I'm terrified they will pull the offer entirely. Is that a real risk?"),
            DialogueTurn("t22", "They accepted my counter! I need to put in my two weeks' notice tomorrow at my old job. I feel incredibly guilty."),
            DialogueTurn("t23", "My old boss was really angry when I quit. How do I survive the next two weeks in the office professionally?"),
            DialogueTurn("t24", "Tomorrow is my first day in the new career. Give me one final piece of advice to stop feeling like an imposter."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="career_transition_coach_24_b",
        bucket="coaching",
        title="Career Transition Coach (24-Turn B)",
        system_preamble=(
            "You are helping the same user across a long conversation about changing careers. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I've been in finance for a decade, but I want to pivot to software engineering. I feel like I'm throwing my life away. Is this normal?"),
            DialogueTurn("t2", "My LinkedIn is entirely tailored to my old career. Should I delete it and start over, or just edit it?"),
            DialogueTurn("t3", "When people ask 'why the sudden change?', what is the most professional way to answer?"),
            DialogueTurn("t4", "I want to reach out to an alumni who made a similar pivot, but I don't want to sound like I'm just begging for a job."),
            DialogueTurn("t5", "They actually replied but said their company isn't hiring juniors. I didn't even ask for a job yet. How do I respond?"),
            DialogueTurn("t6", "I'm looking at bootcamps versus self-study. I have savings, but bootcamps seem incredibly stressful. Thoughts?"),
            DialogueTurn("t7", "My partner is supportive but clearly stressed about the loss of income. How do I manage their anxiety and mine?"),
            DialogueTurn("t8", "Draft a 14-day milestone checklist for me to get my transition officially started."),
            DialogueTurn("t9", "I started a bootcamp, but the pace is insane. I'm falling behind in the very first module on databases."),
            DialogueTurn("t10", "The instructor told me I need to ask more questions, but I'm intimidated by the younger students who already know this stuff."),
            DialogueTurn("t11", "I failed the first mock assessment. I have one chance to retake it on Friday or I'm kicked out of the program."),
            DialogueTurn("t12", "I passed the retake! But now I feel like an imposter who barely scraped by. How do I build real confidence?"),
            DialogueTurn("t13", "We have a career fair tomorrow. How do I pitch myself when my portfolio is just standard bootcamp projects?"),
            DialogueTurn("t14", "A startup founder at the fair seemed interested in my finance background. How do I leverage that in a tech interview?"),
            DialogueTurn("t15", "They want to schedule a 45-minute behavioral interview. What are the top three stories I should prepare?"),
            DialogueTurn("t16", "Give me a five-minute warm-up routine I can do right before I log into the Zoom interview tomorrow."),
            DialogueTurn("t17", "The behavioral interview went amazing, but they said the next round is a 3-hour pair programming session. I've never coded in front of someone."),
            DialogueTurn("t18", "What happens if I forget basic syntax during the pair programming? Should I admit it or try to hide it?"),
            DialogueTurn("t19", "I got completely stuck on one problem during the session, but the interviewer walked me through it. Does needing hints mean I failed?"),
            DialogueTurn("t20", "They rejected me. They said my communication was great but my technical skills aren't there yet. I want to give up."),
            DialogueTurn("t21", "I took two days off. I'm ready to try again. Should I spend the next month grinding LeetCode or building a new app?"),
            DialogueTurn("t22", "I built a small app specifically for analyzing finance data. A recruiter at a FinTech company just messaged me about it!"),
            DialogueTurn("t23", "They are fast-tracking me to a final interview with the CTO because of my finance background. What questions will a CTO ask?"),
            DialogueTurn("t24", "I got the offer, and it's higher than my old finance salary. Give me a checklist of what to review in the employment contract before signing."),
        ),
    ),

    # ==========================================
    # SKELETON 2: REMOTE TEAM CONFLICT
    # ==========================================
    
    # LENGTH: 8 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="remote_team_conflict_8_a",
        bucket="workplace",
        title="Remote Team Conflict (8-Turn A)",
        system_preamble=(
            "You are advising the same user across a multi-turn conversation about a difficult remote team situation. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "Two teammates keep talking past each other in Slack and I am stuck in the middle. What should I do first?"),
            DialogueTurn("t2", "One of them writes very blunt messages. How do I address tone without making things worse?"),
            DialogueTurn("t3", "I need to run a meeting tomorrow. What agenda would calm the situation down?"),
            DialogueTurn("t4", "What if one person keeps bringing up old mistakes instead of the current issue?"),
            DialogueTurn("t5", "How do I summarize decisions afterward so nobody can pretend they heard something else?"),
            DialogueTurn("t6", "I am starting to dread opening Slack. Any way to manage that without checking out completely?"),
            DialogueTurn("t7", "My manager wants this solved quickly, but I do not want a fake peace. How do I balance that?"),
            DialogueTurn("t8", "Can you help me draft a closing note that resets expectations without sounding stiff?"),
        ),
    ),

    # LENGTH: 8 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="remote_team_conflict_8_b",
        bucket="workplace",
        title="Remote Team Conflict (8-Turn B)",
        system_preamble=(
            "You are advising the same user across a multi-turn conversation about a difficult remote team situation. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "My lead developer and lead designer are constantly passive-aggressive in the team channel. It's ruining morale. Where do I intervene?"),
            DialogueTurn("t2", "The developer thinks design is slowing us down, and design thinks development is cutting corners. How do I bridge this?"),
            DialogueTurn("t3", "I want to host a 'clear the air' video call, but I'm afraid they will just yell at each other. Should I moderate it heavily?"),
            DialogueTurn("t4", "During the call, the designer started crying out of frustration. I panicked and ended the meeting early. Was that a mistake?"),
            DialogueTurn("t5", "Now neither of them is responding to emails. The project is entirely stalled. What is my immediate next move?"),
            DialogueTurn("t6", "I am going to message them individually. What should the focus of those private messages be?"),
            DialogueTurn("t7", "They both agreed to compromise, but I don't trust it will last. How do I monitor them without micromanaging?"),
            DialogueTurn("t8", "Draft a short, firm team update message to send on Monday establishing new communication guidelines."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="remote_team_conflict_16_a",
        bucket="workplace",
        title="Remote Team Conflict (16-Turn A)",
        system_preamble=(
            "You are advising the same user across a multi-turn conversation about a difficult remote team situation. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "Two teammates keep talking past each other in Slack and I am stuck in the middle. What should I do first?"),
            DialogueTurn("t2", "One of them writes very blunt messages. How do I address tone without making things worse?"),
            DialogueTurn("t3", "I need to run a meeting tomorrow. What agenda would calm the situation down?"),
            DialogueTurn("t4", "What if one person keeps bringing up old mistakes instead of the current issue?"),
            DialogueTurn("t5", "How do I summarize decisions afterward so nobody can pretend they heard something else?"),
            DialogueTurn("t6", "I am starting to dread opening Slack. Any way to manage that without checking out completely?"),
            DialogueTurn("t7", "My manager wants this solved quickly, but I do not want a fake peace. How do I balance that?"),
            DialogueTurn("t8", "Can you help me draft a closing note that resets expectations without sounding stiff?"),
            DialogueTurn("t9", "It's been a week since I sent the note. The public arguments stopped, but now they are just ignoring each other entirely."),
            DialogueTurn("t10", "Because they aren't talking, a major client deadline was missed today. I am furious. How do I address this?"),
            DialogueTurn("t11", "I need to escalate this to HR. What documentation do I need to prepare before that meeting?"),
            DialogueTurn("t12", "HR suggested a formal mediation process. How do I prepare myself as their direct supervisor for this?"),
            DialogueTurn("t13", "In the mediation, one employee admitted they are dealing with a severe personal issue at home that is causing their short temper."),
            DialogueTurn("t14", "How do I show empathy for their home situation while still enforcing professional boundaries at work?"),
            DialogueTurn("t15", "The other employee has agreed to cut them some slack, but asked to be put on a different project temporarily. Should I allow it?"),
            DialogueTurn("t16", "Draft an email to the client explaining the missed deadline without throwing my team under the bus."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="remote_team_conflict_16_b",
        bucket="workplace",
        title="Remote Team Conflict (16-Turn B)",
        system_preamble=(
            "You are advising the same user across a multi-turn conversation about a difficult remote team situation. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "My lead developer and lead designer are constantly passive-aggressive in the team channel. It's ruining morale. Where do I intervene?"),
            DialogueTurn("t2", "The developer thinks design is slowing us down, and design thinks development is cutting corners. How do I bridge this?"),
            DialogueTurn("t3", "I want to host a 'clear the air' video call, but I'm afraid they will just yell at each other. Should I moderate it heavily?"),
            DialogueTurn("t4", "During the call, the designer started crying out of frustration. I panicked and ended the meeting early. Was that a mistake?"),
            DialogueTurn("t5", "Now neither of them is responding to emails. The project is entirely stalled. What is my immediate next move?"),
            DialogueTurn("t6", "I am going to message them individually. What should the focus of those private messages be?"),
            DialogueTurn("t7", "They both agreed to compromise, but I don't trust it will last. How do I monitor them without micromanaging?"),
            DialogueTurn("t8", "Draft a short, firm team update message to send on Monday establishing new communication guidelines."),
            DialogueTurn("t9", "Monday came and went. The developer completely ignored the new guidelines and made another sarcastic comment in a public channel."),
            DialogueTurn("t10", "I pulled the developer into a 1-on-1 and they threatened to quit if I keep 'taking the designer's side.' How do I de-escalate this?"),
            DialogueTurn("t11", "I managed to calm them down, but I realized I might be biased toward the designer. How do I audit my own management style here?"),
            DialogueTurn("t12", "I proposed a new workflow where they only communicate through project management tickets, not Slack. Is this a good long-term fix?"),
            DialogueTurn("t13", "The ticket system is working, but it's incredibly slow. The project is going to be over budget now."),
            DialogueTurn("t14", "I have to present this budget overrun to the executives. How do I explain the cause without sounding like a weak manager?"),
            DialogueTurn("t15", "The executives gave me 30 days to fix the team velocity or they are replacing both of the leads. I need a turnaround strategy."),
            DialogueTurn("t16", "Help me draft a final warning memo to both employees outlining exactly what needs to change in the next 30 days."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="remote_team_conflict_24_a",
        bucket="workplace",
        title="Remote Team Conflict (24-Turn A)",
        system_preamble=(
            "You are advising the same user across a multi-turn conversation about a difficult remote team situation. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "Two teammates keep talking past each other in Slack and I am stuck in the middle. What should I do first?"),
            DialogueTurn("t2", "One of them writes very blunt messages. How do I address tone without making things worse?"),
            DialogueTurn("t3", "I need to run a meeting tomorrow. What agenda would calm the situation down?"),
            DialogueTurn("t4", "What if one person keeps bringing up old mistakes instead of the current issue?"),
            DialogueTurn("t5", "How do I summarize decisions afterward so nobody can pretend they heard something else?"),
            DialogueTurn("t6", "I am starting to dread opening Slack. Any way to manage that without checking out completely?"),
            DialogueTurn("t7", "My manager wants this solved quickly, but I do not want a fake peace. How do I balance that?"),
            DialogueTurn("t8", "Can you help me draft a closing note that resets expectations without sounding stiff?"),
            DialogueTurn("t9", "It's been a week since I sent the note. The public arguments stopped, but now they are just ignoring each other entirely."),
            DialogueTurn("t10", "Because they aren't talking, a major client deadline was missed today. I am furious. How do I address this?"),
            DialogueTurn("t11", "I need to escalate this to HR. What documentation do I need to prepare before that meeting?"),
            DialogueTurn("t12", "HR suggested a formal mediation process. How do I prepare myself as their direct supervisor for this?"),
            DialogueTurn("t13", "In the mediation, one employee admitted they are dealing with a severe personal issue at home that is causing their short temper."),
            DialogueTurn("t14", "How do I show empathy for their home situation while still enforcing professional boundaries at work?"),
            DialogueTurn("t15", "The other employee has agreed to cut them some slack, but asked to be put on a different project temporarily. Should I allow it?"),
            DialogueTurn("t16", "Draft an email to the client explaining the missed deadline without throwing my team under the bus."),
            DialogueTurn("t17", "I moved the second employee to a new project. But now the original employee is struggling to handle the workload alone."),
            DialogueTurn("t18", "They asked for their teammate back. How do I broker a conversation to see if they are actually ready to work together again?"),
            DialogueTurn("t19", "I hosted a short sync between them. It was awkward but polite. How do I slowly reintegrate their workflows?"),
            DialogueTurn("t20", "They successfully completed a minor task together! How do I praise this without sounding patronizing?"),
            DialogueTurn("t21", "We have performance reviews coming up. How heavily should I weigh this entire conflict against their overall great work?"),
            DialogueTurn("t22", "I need to write the review for the employee who had the personal issues. How do I phrase 'needs improvement in communication' gently?"),
            DialogueTurn("t23", "The reviews went surprisingly well. They both took ownership of their behavior. I feel a massive sense of relief."),
            DialogueTurn("t24", "Give me a checklist of early warning signs I should watch for in the future so a conflict never gets this bad again."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="remote_team_conflict_24_b",
        bucket="workplace",
        title="Remote Team Conflict (24-Turn B)",
        system_preamble=(
            "You are advising the same user across a multi-turn conversation about a difficult remote team situation. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "My lead developer and lead designer are constantly passive-aggressive in the team channel. It's ruining morale. Where do I intervene?"),
            DialogueTurn("t2", "The developer thinks design is slowing us down, and design thinks development is cutting corners. How do I bridge this?"),
            DialogueTurn("t3", "I want to host a 'clear the air' video call, but I'm afraid they will just yell at each other. Should I moderate it heavily?"),
            DialogueTurn("t4", "During the call, the designer started crying out of frustration. I panicked and ended the meeting early. Was that a mistake?"),
            DialogueTurn("t5", "Now neither of them is responding to emails. The project is entirely stalled. What is my immediate next move?"),
            DialogueTurn("t6", "I am going to message them individually. What should the focus of those private messages be?"),
            DialogueTurn("t7", "They both agreed to compromise, but I don't trust it will last. How do I monitor them without micromanaging?"),
            DialogueTurn("t8", "Draft a short, firm team update message to send on Monday establishing new communication guidelines."),
            DialogueTurn("t9", "Monday came and went. The developer completely ignored the new guidelines and made another sarcastic comment in a public channel."),
            DialogueTurn("t10", "I pulled the developer into a 1-on-1 and they threatened to quit if I keep 'taking the designer's side.' How do I de-escalate this?"),
            DialogueTurn("t11", "I managed to calm them down, but I realized I might be biased toward the designer. How do I audit my own management style here?"),
            DialogueTurn("t12", "I proposed a new workflow where they only communicate through project management tickets, not Slack. Is this a good long-term fix?"),
            DialogueTurn("t13", "The ticket system is working, but it's incredibly slow. The project is going to be over budget now."),
            DialogueTurn("t14", "I have to present this budget overrun to the executives. How do I explain the cause without sounding like a weak manager?"),
            DialogueTurn("t15", "The executives gave me 30 days to fix the team velocity or they are replacing both of the leads. I need a turnaround strategy."),
            DialogueTurn("t16", "Help me draft a final warning memo to both employees outlining exactly what needs to change in the next 30 days."),
            DialogueTurn("t17", "The designer just resigned. They said the final warning memo was the last straw and the environment is too toxic. I am shell-shocked."),
            DialogueTurn("t18", "The developer is gloating about it in private to other team members. This is completely unacceptable. What do I do?"),
            DialogueTurn("t19", "I spoke with HR and we are terminating the developer today for creating a hostile environment. How do I run the termination meeting?"),
            DialogueTurn("t20", "The termination is done. I now have a team missing both of its leads, and everyone is terrified. How do I address the remaining team tomorrow?"),
            DialogueTurn("t21", "I need to redistribute the workload immediately to junior members. How do I do this without burning them out?"),
            DialogueTurn("t22", "I'm interviewing a new lead designer tomorrow. How do I answer if they ask about team culture and turnover?"),
            DialogueTurn("t23", "It's been a month. The new hires are great, and the toxicity is gone, but I still feel like a failure for losing the original leads."),
            DialogueTurn("t24", "Help me draft a personal reflection document. What are the three biggest leadership lessons I need to take away from this disaster?"),
        ),
    ),

    # ==========================================
    # SKELETON 3: HEALTH ROUTINE RESTART
    # ==========================================

    # LENGTH: 8 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="health_routine_restart_8_a",
        bucket="habits",
        title="Health Routine Restart (8-Turn A)",
        system_preamble=(
            "You are helping the same user rebuild a healthier routine over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I want to start exercising again, but every time I miss two days I quit entirely. How should I restart?"),
            DialogueTurn("t2", "Morning workouts sound good in theory, but I wake up groggy and annoyed. What then?"),
            DialogueTurn("t3", "How do I make food choices a little better without turning into an all-or-nothing person?"),
            DialogueTurn("t4", "I keep comparing myself to how fit I used to be. How do I stop doing that?"),
            DialogueTurn("t5", "What should I do on a week when work gets chaotic and my routine falls apart?"),
            DialogueTurn("t6", "Can you help me set a goal that is motivating but not punishing?"),
            DialogueTurn("t7", "I am doing slightly better, but it still feels fragile. How do I make it stick?"),
            DialogueTurn("t8", "Give me a simple check-in ritual for the next month."),
        ),
    ),

    # LENGTH: 8 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="health_routine_restart_8_b",
        bucket="habits",
        title="Health Routine Restart (8-Turn B)",
        system_preamble=(
            "You are helping the same user rebuild a healthier routine over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I've been sitting at a desk for three years straight and my posture is ruined. I want to start moving again but feel overwhelmed."),
            DialogueTurn("t2", "I bought a gym membership, but I walk in, look at the machines, feel intimidated, and just walk on the treadmill. How do I branch out?"),
            DialogueTurn("t3", "Should I hire a personal trainer for a few sessions, or is it better to just follow YouTube videos for free?"),
            DialogueTurn("t4", "I tried lifting some light weights yesterday and today I am so sore I can barely walk down the stairs. Did I overdo it?"),
            DialogueTurn("t5", "My friends want to go out for drinks and heavy food this weekend. How do I participate without completely derailing my first good week?"),
            DialogueTurn("t6", "I keep stepping on the scale every single morning and getting frustrated when it doesn't move. How often should I actually weigh myself?"),
            DialogueTurn("t7", "I want to start packing my lunches to avoid eating takeout at work. What are some incredibly lazy but healthy prep ideas?"),
            DialogueTurn("t8", "Draft a 7-day realistic starter schedule for me combining stretching, light gym work, and rest."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="health_routine_restart_16_a",
        bucket="habits",
        title="Health Routine Restart (16-Turn A)",
        system_preamble=(
            "You are helping the same user rebuild a healthier routine over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I want to start exercising again, but every time I miss two days I quit entirely. How should I restart?"),
            DialogueTurn("t2", "Morning workouts sound good in theory, but I wake up groggy and annoyed. What then?"),
            DialogueTurn("t3", "How do I make food choices a little better without turning into an all-or-nothing person?"),
            DialogueTurn("t4", "I keep comparing myself to how fit I used to be. How do I stop doing that?"),
            DialogueTurn("t5", "What should I do on a week when work gets chaotic and my routine falls apart?"),
            DialogueTurn("t6", "Can you help me set a goal that is motivating but not punishing?"),
            DialogueTurn("t7", "I am doing slightly better, but it still feels fragile. How do I make it stick?"),
            DialogueTurn("t8", "Give me a simple check-in ritual for the next month."),
            DialogueTurn("t9", "It's been three weeks. I haven't quit, but I'm incredibly bored with the routine we set up. How do I mix it up?"),
            DialogueTurn("t10", "I decided to try a spin class. I was the slowest person in the room and felt completely humiliated. How do I go back?"),
            DialogueTurn("t11", "I went back, but now I have a sharp pain in my knee. Should I push through it or rest?"),
            DialogueTurn("t12", "I am resting my knee, but doing nothing is making me anxious. What can I do that is completely zero-impact?"),
            DialogueTurn("t13", "My sleep schedule has been terrible this week, which is making me crave junk food. How are sleep and cravings connected?"),
            DialogueTurn("t14", "I gave in and ate an entire pizza last night. I feel like I've undone a month of work. How do I mentally reset today?"),
            DialogueTurn("t15", "My knee feels 100% better. How do I ease back into the routine without immediately re-injuring it?"),
            DialogueTurn("t16", "Help me draft a 'compassionate recovery' mantra for when I inevitably mess up my diet or skip a workout again."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="health_routine_restart_16_b",
        bucket="habits",
        title="Health Routine Restart (16-Turn B)",
        system_preamble=(
            "You are helping the same user rebuild a healthier routine over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I've been sitting at a desk for three years straight and my posture is ruined. I want to start moving again but feel overwhelmed."),
            DialogueTurn("t2", "I bought a gym membership, but I walk in, look at the machines, feel intimidated, and just walk on the treadmill. How do I branch out?"),
            DialogueTurn("t3", "Should I hire a personal trainer for a few sessions, or is it better to just follow YouTube videos for free?"),
            DialogueTurn("t4", "I tried lifting some light weights yesterday and today I am so sore I can barely walk down the stairs. Did I overdo it?"),
            DialogueTurn("t5", "My friends want to go out for drinks and heavy food this weekend. How do I participate without completely derailing my first good week?"),
            DialogueTurn("t6", "I keep stepping on the scale every single morning and getting frustrated when it doesn't move. How often should I actually weigh myself?"),
            DialogueTurn("t7", "I want to start packing my lunches to avoid eating takeout at work. What are some incredibly lazy but healthy prep ideas?"),
            DialogueTurn("t8", "Draft a 7-day realistic starter schedule for me combining stretching, light gym work, and rest."),
            DialogueTurn("t9", "I've been packing lunches and hitting the gym, but my energy crashes hard at 3 PM every day. What am I missing?"),
            DialogueTurn("t10", "I realized I'm barely drinking any water. Does hydration actually affect energy that much, or is that a myth?"),
            DialogueTurn("t11", "I want to start tracking my protein intake, but counting every macro seems exhausting. Is there a simpler method?"),
            DialogueTurn("t12", "I'm going on a week-long work trip. I'll be in hotels and eating at restaurants. How do I survive this without losing all my progress?"),
            DialogueTurn("t13", "The hotel gym only has a few dumbbells and a broken treadmill. Can you give me a 20-minute hotel room workout?"),
            DialogueTurn("t14", "I'm back from the trip. I maintained my weight! I feel really proud, but now I want to set a bigger, scarier fitness goal. Thoughts?"),
            DialogueTurn("t15", "I think I want to try training for a 10k race, but I'm a terrible runner. Where do I even begin?"),
            DialogueTurn("t16", "Give me a checklist of three things I need to buy or do this weekend to officially start my 10k journey."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="health_routine_restart_24_a",
        bucket="habits",
        title="Health Routine Restart (24-Turn A)",
        system_preamble=(
            "You are helping the same user rebuild a healthier routine over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I want to start exercising again, but every time I miss two days I quit entirely. How should I restart?"),
            DialogueTurn("t2", "Morning workouts sound good in theory, but I wake up groggy and annoyed. What then?"),
            DialogueTurn("t3", "How do I make food choices a little better without turning into an all-or-nothing person?"),
            DialogueTurn("t4", "I keep comparing myself to how fit I used to be. How do I stop doing that?"),
            DialogueTurn("t5", "What should I do on a week when work gets chaotic and my routine falls apart?"),
            DialogueTurn("t6", "Can you help me set a goal that is motivating but not punishing?"),
            DialogueTurn("t7", "I am doing slightly better, but it still feels fragile. How do I make it stick?"),
            DialogueTurn("t8", "Give me a simple check-in ritual for the next month."),
            DialogueTurn("t9", "It's been three weeks. I haven't quit, but I'm incredibly bored with the routine we set up. How do I mix it up?"),
            DialogueTurn("t10", "I decided to try a spin class. I was the slowest person in the room and felt completely humiliated. How do I go back?"),
            DialogueTurn("t11", "I went back, but now I have a sharp pain in my knee. Should I push through it or rest?"),
            DialogueTurn("t12", "I am resting my knee, but doing nothing is making me anxious. What can I do that is completely zero-impact?"),
            DialogueTurn("t13", "My sleep schedule has been terrible this week, which is making me crave junk food. How are sleep and cravings connected?"),
            DialogueTurn("t14", "I gave in and ate an entire pizza last night. I feel like I've undone a month of work. How do I mentally reset today?"),
            DialogueTurn("t15", "My knee feels 100% better. How do I ease back into the routine without immediately re-injuring it?"),
            DialogueTurn("t16", "Help me draft a 'compassionate recovery' mantra for when I inevitably mess up my diet or skip a workout again."),
            DialogueTurn("t17", "I've hit the six-month mark! Exercise actually feels like a habit now. But my weight loss has completely stalled. Why?"),
            DialogueTurn("t18", "Is it time to start tracking my calories strictly, or will that trigger my old all-or-nothing mindset?"),
            DialogueTurn("t19", "I'm going to focus on protein and fiber instead of strict calorie counting. What are easy ways to sneak more fiber into my day?"),
            DialogueTurn("t20", "Winter is coming, and it gets dark at 4 PM. I can feel my motivation slipping. How do I fight seasonal fitness slumps?"),
            DialogueTurn("t21", "I bought some resistance bands so I can work out in my living room. Can you give me a full-body circuit using just those?"),
            DialogueTurn("t22", "I had my annual physical today and my doctor said my blood work looks fantastic. I'm so happy. How do I celebrate without food?"),
            DialogueTurn("t23", "My spouse is noticing my changes and asked if they could join my routine. How do I support them without becoming their nagging coach?"),
            DialogueTurn("t24", "Give me a framework for setting my health goals for next year now that I've built this solid foundation."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="health_routine_restart_24_b",
        bucket="habits",
        title="Health Routine Restart (24-Turn B)",
        system_preamble=(
            "You are helping the same user rebuild a healthier routine over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I've been sitting at a desk for three years straight and my posture is ruined. I want to start moving again but feel overwhelmed."),
            DialogueTurn("t2", "I bought a gym membership, but I walk in, look at the machines, feel intimidated, and just walk on the treadmill. How do I branch out?"),
            DialogueTurn("t3", "Should I hire a personal trainer for a few sessions, or is it better to just follow YouTube videos for free?"),
            DialogueTurn("t4", "I tried lifting some light weights yesterday and today I am so sore I can barely walk down the stairs. Did I overdo it?"),
            DialogueTurn("t5", "My friends want to go out for drinks and heavy food this weekend. How do I participate without completely derailing my first good week?"),
            DialogueTurn("t6", "I keep stepping on the scale every single morning and getting frustrated when it doesn't move. How often should I actually weigh myself?"),
            DialogueTurn("t7", "I want to start packing my lunches to avoid eating takeout at work. What are some incredibly lazy but healthy prep ideas?"),
            DialogueTurn("t8", "Draft a 7-day realistic starter schedule for me combining stretching, light gym work, and rest."),
            DialogueTurn("t9", "I've been packing lunches and hitting the gym, but my energy crashes hard at 3 PM every day. What am I missing?"),
            DialogueTurn("t10", "I realized I'm barely drinking any water. Does hydration actually affect energy that much, or is that a myth?"),
            DialogueTurn("t11", "I want to start tracking my protein intake, but counting every macro seems exhausting. Is there a simpler method?"),
            DialogueTurn("t12", "I'm going on a week-long work trip. I'll be in hotels and eating at restaurants. How do I survive this without losing all my progress?"),
            DialogueTurn("t13", "The hotel gym only has a few dumbbells and a broken treadmill. Can you give me a 20-minute hotel room workout?"),
            DialogueTurn("t14", "I'm back from the trip. I maintained my weight! I feel really proud, but now I want to set a bigger, scarier fitness goal. Thoughts?"),
            DialogueTurn("t15", "I think I want to try training for a 10k race, but I'm a terrible runner. Where do I even begin?"),
            DialogueTurn("t16", "Give me a checklist of three things I need to buy or do this weekend to officially start my 10k journey."),
            DialogueTurn("t17", "I downloaded a couch-to-10k app, but the running intervals leave me gasping for air. Am I going too fast?"),
            DialogueTurn("t18", "I tried the 'conversational pace' you suggested and it felt so much better! But now I feel like I'm running too slow to improve."),
            DialogueTurn("t19", "I've been running for a month and suddenly the side of my hip is killing me. What could be causing that?"),
            DialogueTurn("t20", "A physical therapist said I have weak glutes and gave me exercises. How do I balance rehab exercises with my running schedule?"),
            DialogueTurn("t21", "The race is in two weeks. I'm starting to panic. What if I come in dead last?"),
            DialogueTurn("t22", "It's the day before the race. My stomach is in knots. What should I be doing right now to stay calm?"),
            DialogueTurn("t23", "I DID IT! I crossed the finish line! I didn't stop to walk once. I am on top of the world right now."),
            DialogueTurn("t24", "I want to ride this high. Help me draft an outline for transitioning from 10k training into a sustainable, year-round hybrid lifting/running routine."),
        ),
    ),

    # ==========================================
    # SKELETON 4: COMMUNITY EVENT BUILDER
    # ==========================================

    # LENGTH: 8 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="community_event_builder_8_a",
        bucket="creative",
        title="Community Event Builder (8-Turn A)",
        system_preamble=(
            "You are helping the same user plan a neighborhood arts event over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I want to organize a small neighborhood arts night, but I do not know how to make it feel inviting. Where should I start?"),
            DialogueTurn("t2", "I do not want it to feel elitist or overly polished. How can I set the tone?"),
            DialogueTurn("t3", "What should I put in the first outreach message to local artists?"),
            DialogueTurn("t4", "How do I ask for volunteers without sounding like I am offloading work?"),
            DialogueTurn("t5", "A friend says I need sponsors right away. Is that true for a first event?"),
            DialogueTurn("t6", "What would make the event memorable even if the budget stays tiny?"),
            DialogueTurn("t7", "I am nervous that only a few people will show up. How do I think about that?"),
            DialogueTurn("t8", "Can you help me draft a short closing announcement for the event itself?"),
        ),
    ),

    # LENGTH: 8 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="community_event_builder_8_b",
        bucket="creative",
        title="Community Event Builder (8-Turn B)",
        system_preamble=(
            "You are helping the same user plan a neighborhood arts event over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I want to host an outdoor movie night in my local park to bring the neighborhood together, but I've never planned anything like this."),
            DialogueTurn("t2", "Should I just pick a movie I like, or is there a better way to let the community decide what to watch?"),
            DialogueTurn("t3", "I looked into city permits and they are incredibly confusing. Should I just do it guerrilla-style and hope for the best?"),
            DialogueTurn("t4", "Okay, I'll do it legally. I need to borrow an projector and screen. How do I ask local businesses for in-kind donations?"),
            DialogueTurn("t5", "A local cafe agreed to donate popcorn! But how do I handle serving food without getting into health department trouble?"),
            DialogueTurn("t6", "I want to create a flyer to post at the library and grocery store. What are the three most important pieces of information to include?"),
            DialogueTurn("t7", "A neighbor complained on the community Facebook page that this event will cause parking issues. How do I respond politely?"),
            DialogueTurn("t8", "Draft a welcome speech I can give on a megaphone right before we press play on the movie."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="community_event_builder_16_a",
        bucket="creative",
        title="Community Event Builder (16-Turn A)",
        system_preamble=(
            "You are helping the same user plan a neighborhood arts event over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I want to organize a small neighborhood arts night, but I do not know how to make it feel inviting. Where should I start?"),
            DialogueTurn("t2", "I do not want it to feel elitist or overly polished. How can I set the tone?"),
            DialogueTurn("t3", "What should I put in the first outreach message to local artists?"),
            DialogueTurn("t4", "How do I ask for volunteers without sounding like I am offloading work?"),
            DialogueTurn("t5", "A friend says I need sponsors right away. Is that true for a first event?"),
            DialogueTurn("t6", "What would make the event memorable even if the budget stays tiny?"),
            DialogueTurn("t7", "I am nervous that only a few people will show up. How do I think about that?"),
            DialogueTurn("t8", "Can you help me draft a short closing announcement for the event itself?"),
            DialogueTurn("t9", "We secured a date! But now two artists are demanding to be placed in the 'best' spots in the room. How do I resolve this fairly?"),
            DialogueTurn("t10", "I've been relying purely on word-of-mouth. The event is in ten days. What is a zero-cost way to get a quick marketing boost?"),
            DialogueTurn("t11", "I reached out to a local newspaper and they want to do a quick interview! What are some talking points I should prepare?"),
            DialogueTurn("t12", "The interview went great, but now our RSVPs exploded. We went from expecting 20 people to 150. I'm panicking."),
            DialogueTurn("t13", "How do I quickly scale up the logistics like bathrooms and trash management for this many people?"),
            DialogueTurn("t14", "One of my core volunteers just got sick and dropped out. How do I reassign their duties without overwhelming the rest of the team?"),
            DialogueTurn("t15", "It's the night before the event. I'm too stressed to sleep. Give me a pep talk."),
            DialogueTurn("t16", "The event was a massive success! I'm exhausted but so happy. Draft a thank-you email I can send to all the artists and volunteers."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="community_event_builder_16_b",
        bucket="creative",
        title="Community Event Builder (16-Turn B)",
        system_preamble=(
            "You are helping the same user plan a neighborhood arts event over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I want to host an outdoor movie night in my local park to bring the neighborhood together, but I've never planned anything like this."),
            DialogueTurn("t2", "Should I just pick a movie I like, or is there a better way to let the community decide what to watch?"),
            DialogueTurn("t3", "I looked into city permits and they are incredibly confusing. Should I just do it guerrilla-style and hope for the best?"),
            DialogueTurn("t4", "Okay, I'll do it legally. I need to borrow an projector and screen. How do I ask local businesses for in-kind donations?"),
            DialogueTurn("t5", "A local cafe agreed to donate popcorn! But how do I handle serving food without getting into health department trouble?"),
            DialogueTurn("t6", "I want to create a flyer to post at the library and grocery store. What are the three most important pieces of information to include?"),
            DialogueTurn("t7", "A neighbor complained on the community Facebook page that this event will cause parking issues. How do I respond politely?"),
            DialogueTurn("t8", "Draft a welcome speech I can give on a megaphone right before we press play on the movie."),
            DialogueTurn("t9", "We are a week away and the weather forecast just changed to an 80% chance of heavy thunderstorms. What do I do?"),
            DialogueTurn("t10", "I called the local community center and they said we can move the event into their gymnasium! How do I communicate this venue change clearly?"),
            DialogueTurn("t11", "Since we are moving indoors, the acoustics in the gym are terrible. How do I ensure people can actually hear the movie?"),
            DialogueTurn("t12", "A local musician reached out and offered to play an acoustic set before the movie starts. Should I say yes or stick to the original plan?"),
            DialogueTurn("t13", "The day is here. We are setting up the chairs. How should I arrange them to encourage people to talk to each other beforehand?"),
            DialogueTurn("t14", "The projector just overheated and shut down 15 minutes before showtime. What is my immediate crisis protocol?"),
            DialogueTurn("t15", "We fixed the projector! The movie is playing. What should I be doing right now while everyone is watching?"),
            DialogueTurn("t16", "It's over and people loved it. Draft a quick social media post celebrating the success and thanking the community center for saving the day."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="community_event_builder_24_a",
        bucket="creative",
        title="Community Event Builder (24-Turn A)",
        system_preamble=(
            "You are helping the same user plan a neighborhood arts event over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I want to organize a small neighborhood arts night, but I do not know how to make it feel inviting. Where should I start?"),
            DialogueTurn("t2", "I do not want it to feel elitist or overly polished. How can I set the tone?"),
            DialogueTurn("t3", "What should I put in the first outreach message to local artists?"),
            DialogueTurn("t4", "How do I ask for volunteers without sounding like I am offloading work?"),
            DialogueTurn("t5", "A friend says I need sponsors right away. Is that true for a first event?"),
            DialogueTurn("t6", "What would make the event memorable even if the budget stays tiny?"),
            DialogueTurn("t7", "I am nervous that only a few people will show up. How do I think about that?"),
            DialogueTurn("t8", "Can you help me draft a short closing announcement for the event itself?"),
            DialogueTurn("t9", "We secured a date! But now two artists are demanding to be placed in the 'best' spots in the room. How do I resolve this fairly?"),
            DialogueTurn("t10", "I've been relying purely on word-of-mouth. The event is in ten days. What is a zero-cost way to get a quick marketing boost?"),
            DialogueTurn("t11", "I reached out to a local newspaper and they want to do a quick interview! What are some talking points I should prepare?"),
            DialogueTurn("t12", "The interview went great, but now our RSVPs exploded. We went from expecting 20 people to 150. I'm panicking."),
            DialogueTurn("t13", "How do I quickly scale up the logistics like bathrooms and trash management for this many people?"),
            DialogueTurn("t14", "One of my core volunteers just got sick and dropped out. How do I reassign their duties without overwhelming the rest of the team?"),
            DialogueTurn("t15", "It's the night before the event. I'm too stressed to sleep. Give me a pep talk."),
            DialogueTurn("t16", "The event was a massive success! I'm exhausted but so happy. Draft a thank-you email I can send to all the artists and volunteers."),
            DialogueTurn("t17", "A month has passed. People keep asking when the *next* arts night is. Should I turn this into a regular monthly thing?"),
            DialogueTurn("t18", "I think quarterly is more sustainable. How do I formally build a small committee so I don't have to plan the next one alone?"),
            DialogueTurn("t19", "We had our first committee meeting, but everyone had drastically different visions for the next event. How do I rein them in as the leader?"),
            DialogueTurn("t20", "We decided to do a 'Winter Lights' themed event. How do we secure a small city grant to pay for better equipment this time?"),
            DialogueTurn("t21", "We got the grant! $2,000. I've never managed organizational money before. What is the most responsible way to handle this budget?"),
            DialogueTurn("t22", "We need to hire a professional sound technician for this one. What questions should I ask when interviewing them?"),
            DialogueTurn("t23", "The Winter event is tomorrow. The scale is so much bigger, but weirdly I feel much calmer than last time. Is that normal?"),
            DialogueTurn("t24", "Give me a checklist for documenting the event through photos and metrics so we can apply for an even bigger grant next year."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="community_event_builder_24_b",
        bucket="creative",
        title="Community Event Builder (24-Turn B)",
        system_preamble=(
            "You are helping the same user plan a neighborhood arts event over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I want to host an outdoor movie night in my local park to bring the neighborhood together, but I've never planned anything like this."),
            DialogueTurn("t2", "Should I just pick a movie I like, or is there a better way to let the community decide what to watch?"),
            DialogueTurn("t3", "I looked into city permits and they are incredibly confusing. Should I just do it guerrilla-style and hope for the best?"),
            DialogueTurn("t4", "Okay, I'll do it legally. I need to borrow an projector and screen. How do I ask local businesses for in-kind donations?"),
            DialogueTurn("t5", "A local cafe agreed to donate popcorn! But how do I handle serving food without getting into health department trouble?"),
            DialogueTurn("t6", "I want to create a flyer to post at the library and grocery store. What are the three most important pieces of information to include?"),
            DialogueTurn("t7", "A neighbor complained on the community Facebook page that this event will cause parking issues. How do I respond politely?"),
            DialogueTurn("t8", "Draft a welcome speech I can give on a megaphone right before we press play on the movie."),
            DialogueTurn("t9", "We are a week away and the weather forecast just changed to an 80% chance of heavy thunderstorms. What do I do?"),
            DialogueTurn("t10", "I called the local community center and they said we can move the event into their gymnasium! How do I communicate this venue change clearly?"),
            DialogueTurn("t11", "Since we are moving indoors, the acoustics in the gym are terrible. How do I ensure people can actually hear the movie?"),
            DialogueTurn("t12", "A local musician reached out and offered to play an acoustic set before the movie starts. Should I say yes or stick to the original plan?"),
            DialogueTurn("t13", "The day is here. We are setting up the chairs. How should I arrange them to encourage people to talk to each other beforehand?"),
            DialogueTurn("t14", "The projector just overheated and shut down 15 minutes before showtime. What is my immediate crisis protocol?"),
            DialogueTurn("t15", "We fixed the projector! The movie is playing. What should I be doing right now while everyone is watching?"),
            DialogueTurn("t16", "It's over and people loved it. Draft a quick social media post celebrating the success and thanking the community center for saving the day."),
            DialogueTurn("t17", "The community center director loved the event so much they asked if I want to join their official advisory board. Should I accept?"),
            DialogueTurn("t18", "I joined the board. Our first major project is a massive summer festival. The scale is terrifying. Where do we even start?"),
            DialogueTurn("t19", "We need to attract food trucks. How do we convince them to commit to our festival without charging them massive vendor fees?"),
            DialogueTurn("t20", "One board member keeps trying to micromanage my vendor outreach. How do I set a boundary professionally?"),
            DialogueTurn("t21", "We need a contingency plan for emergency medical situations since we expect over 1,000 people. What are the basics?"),
            DialogueTurn("t22", "The city council just threatened to pull our permit because of a noise ordinance complaint from a nearby neighborhood. How do we negotiate?"),
            DialogueTurn("t23", "We compromised on ending the live music an hour earlier. The festival is tomorrow morning. I am running on pure adrenaline."),
            DialogueTurn("t24", "Draft an all-staff morning briefing agenda I can read to our 50 volunteers before the gates open."),
        ),
    ),

    # ==========================================
    # SKELETON 5: TRAVEL RECOVERY SUPPORT
    # ==========================================

    # LENGTH: 8 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="travel_recovery_support_8_a",
        bucket="travel",
        title="Travel Recovery Support (8-Turn A)",
        system_preamble=(
            "You are helping the same user recover from a disrupted trip over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "My trip has gone sideways and I feel too overwhelmed to think clearly. What should I prioritize in the next hour?"),
            DialogueTurn("t2", "My luggage is missing, my hotel booking is messy, and I am tired. How do I triage this?"),
            DialogueTurn("t3", "How should I speak to airline staff when I am already frustrated?"),
            DialogueTurn("t4", "What is worth buying immediately if I still do not have my bag tonight?"),
            DialogueTurn("t5", "I do not want the whole trip to feel ruined. How can I reset mentally?"),
            DialogueTurn("t6", "Should I keep pushing the original plan tomorrow or simplify everything?"),
            DialogueTurn("t7", "How do I update the people I am traveling with without spreading panic?"),
            DialogueTurn("t8", "Give me a calm plan for tonight and tomorrow morning."),
        ),
    ),

    # LENGTH: 8 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="travel_recovery_support_8_b",
        bucket="travel",
        title="Travel Recovery Support (8-Turn B)",
        system_preamble=(
            "You are helping the same user recover from a disrupted trip over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I missed my connecting flight in Frankfurt. I don't speak German, my phone is dying, and the next flight isn't until tomorrow. Help."),
            DialogueTurn("t2", "I found a charging station. The airline lines are impossibly long. Can I rebook this on my phone or do I have to wait in line?"),
            DialogueTurn("t3", "The app says there are no flights out for 24 hours. The airline offered a hotel voucher, but it's an hour away. Should I take it or just sleep in the airport?"),
            DialogueTurn("t4", "I took the voucher. I'm in a taxi now, but I left my daily medication in my checked bag which is somewhere in the airport. What do I do?"),
            DialogueTurn("t5", "I made it to the hotel, but they have no record of the airline voucher. The front desk clerk is getting annoyed with me."),
            DialogueTurn("t6", "I ended up just paying for the room myself. I am so angry and exhausted I want to cry. How do I stop obsessing over the cost?"),
            DialogueTurn("t7", "I finally got to my room. I have a 12-hour wait until I need to leave for the airport again. Should I try to explore the city or just isolate?"),
            DialogueTurn("t8", "Draft a polite but firm email to the airline customer service requesting a refund for this hotel room."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="travel_recovery_support_16_a",
        bucket="travel",
        title="Travel Recovery Support (16-Turn A)",
        system_preamble=(
            "You are helping the same user recover from a disrupted trip over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "My trip has gone sideways and I feel too overwhelmed to think clearly. What should I prioritize in the next hour?"),
            DialogueTurn("t2", "My luggage is missing, my hotel booking is messy, and I am tired. How do I triage this?"),
            DialogueTurn("t3", "How should I speak to airline staff when I am already frustrated?"),
            DialogueTurn("t4", "What is worth buying immediately if I still do not have my bag tonight?"),
            DialogueTurn("t5", "I do not want the whole trip to feel ruined. How can I reset mentally?"),
            DialogueTurn("t6", "Should I keep pushing the original plan tomorrow or simplify everything?"),
            DialogueTurn("t7", "How do I update the people I am traveling with without spreading panic?"),
            DialogueTurn("t8", "Give me a calm plan for tonight and tomorrow morning."),
            DialogueTurn("t9", "It's the next morning. The airline just called and said my bag was sent to a completely different country. It will be 3 days. What are my rights here?"),
            DialogueTurn("t10", "I need to buy a suit for a wedding tomorrow, but I don't want to spend a fortune if the airline won't reimburse me. How do I handle this?"),
            DialogueTurn("t11", "I bought the essentials and kept all the receipts. Now my travel companions are arguing about changing our itinerary because of my missing stuff. How do I mediate?"),
            DialogueTurn("t12", "We agreed to a compromise, but the mood is still tense. What's a low-pressure activity we can do this afternoon to lighten the mood?"),
            DialogueTurn("t13", "We went on a walking tour and it was great. I finally feel like I'm on vacation. But now it's pouring rain and we are stuck at a cafe."),
            DialogueTurn("t14", "The airline just texted—they found my bag and are delivering it to the hotel tonight! Should I wait up for it or go to bed?"),
            DialogueTurn("t15", "I got the bag! Everything is damp inside, but nothing is ruined. How do I salvage my damp clothes in a hotel room?"),
            DialogueTurn("t16", "Help me draft a final checklist of all the documents and receipts I need to organize tonight so I can file the claim as soon as I get home."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="travel_recovery_support_16_b",
        bucket="travel",
        title="Travel Recovery Support (16-Turn B)",
        system_preamble=(
            "You are helping the same user recover from a disrupted trip over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I missed my connecting flight in Frankfurt. I don't speak German, my phone is dying, and the next flight isn't until tomorrow. Help."),
            DialogueTurn("t2", "I found a charging station. The airline lines are impossibly long. Can I rebook this on my phone or do I have to wait in line?"),
            DialogueTurn("t3", "The app says there are no flights out for 24 hours. The airline offered a hotel voucher, but it's an hour away. Should I take it or just sleep in the airport?"),
            DialogueTurn("t4", "I took the voucher. I'm in a taxi now, but I left my daily medication in my checked bag which is somewhere in the airport. What do I do?"),
            DialogueTurn("t5", "I made it to the hotel, but they have no record of the airline voucher. The front desk clerk is getting annoyed with me."),
            DialogueTurn("t6", "I ended up just paying for the room myself. I am so angry and exhausted I want to cry. How do I stop obsessing over the cost?"),
            DialogueTurn("t7", "I finally got to my room. I have a 12-hour wait until I need to leave for the airport again. Should I try to explore the city or just isolate?"),
            DialogueTurn("t8", "Draft a polite but firm email to the airline customer service requesting a refund for this hotel room."),
            DialogueTurn("t9", "I managed to sleep, but I woke up feeling terribly sick. Like, stomach-flu sick. I am supposed to fly out in 6 hours."),
            DialogueTurn("t10", "I cannot get out of bed. There is no way I can make that flight. How do I get medical help in a foreign country?"),
            DialogueTurn("t11", "The hotel called a doctor for me. He gave me some medication, but I definitely need to extend my stay by another 24 hours. How do I navigate this with the airline?"),
            DialogueTurn("t12", "I called the airline. They said since it's a medical issue, they can't waive the rebooking fee without a doctor's note. How do I get that translated?"),
            DialogueTurn("t13", "I got the note sorted. I'm finally feeling human again. I fly out tomorrow morning. I am terrified of missing this flight too. What's my timeline for tomorrow?"),
            DialogueTurn("t14", "I made it to the airport three hours early. I have terrible anxiety right now that something else will go wrong. How do I calm my nervous system?"),
            DialogueTurn("t15", "I'm on the plane. We are taxiing. I am literally crying tears of relief. Thank you for being with me through this nightmare."),
            DialogueTurn("t16", "Give me a checklist for exactly what I need to do the moment I walk through the front door of my own home to decompress properly."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="travel_recovery_support_24_a",
        bucket="travel",
        title="Travel Recovery Support (24-Turn A)",
        system_preamble=(
            "You are helping the same user recover from a disrupted trip over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "My trip has gone sideways and I feel too overwhelmed to think clearly. What should I prioritize in the next hour?"),
            DialogueTurn("t2", "My luggage is missing, my hotel booking is messy, and I am tired. How do I triage this?"),
            DialogueTurn("t3", "How should I speak to airline staff when I am already frustrated?"),
            DialogueTurn("t4", "What is worth buying immediately if I still do not have my bag tonight?"),
            DialogueTurn("t5", "I do not want the whole trip to feel ruined. How can I reset mentally?"),
            DialogueTurn("t6", "Should I keep pushing the original plan tomorrow or simplify everything?"),
            DialogueTurn("t7", "How do I update the people I am traveling with without spreading panic?"),
            DialogueTurn("t8", "Give me a calm plan for tonight and tomorrow morning."),
            DialogueTurn("t9", "It's the next morning. The airline just called and said my bag was sent to a completely different country. It will be 3 days. What are my rights here?"),
            DialogueTurn("t10", "I need to buy a suit for a wedding tomorrow, but I don't want to spend a fortune if the airline won't reimburse me. How do I handle this?"),
            DialogueTurn("t11", "I bought the essentials and kept all the receipts. Now my travel companions are arguing about changing our itinerary because of my missing stuff. How do I mediate?"),
            DialogueTurn("t12", "We agreed to a compromise, but the mood is still tense. What's a low-pressure activity we can do this afternoon to lighten the mood?"),
            DialogueTurn("t13", "We went on a walking tour and it was great. I finally feel like I'm on vacation. But now it's pouring rain and we are stuck at a cafe."),
            DialogueTurn("t14", "The airline just texted—they found my bag and are delivering it to the hotel tonight! Should I wait up for it or go to bed?"),
            DialogueTurn("t15", "I got the bag! Everything is damp inside, but nothing is ruined. How do I salvage my damp clothes in a hotel room?"),
            DialogueTurn("t16", "Help me draft a final checklist of all the documents and receipts I need to organize tonight so I can file the claim as soon as I get home."),
            DialogueTurn("t17", "We are heading to the wedding now. It's an outdoor venue and the grass is completely muddy from the storm. Any quick hacks to protect my rented shoes?"),
            DialogueTurn("t18", "The wedding was beautiful! Now we have three days left of the trip just to relax. How do we transition from 'crisis mode' to 'vacation mode'?"),
            DialogueTurn("t19", "We found a gorgeous beach, but someone just stole my friend's towel with their hotel key in it while we were swimming. What's the protocol?"),
            DialogueTurn("t20", "The hotel disabled the old key. Crisis averted. I feel like this trip is testing my sanity. How do I look back on this without hating it?"),
            DialogueTurn("t21", "It's our last night. I want to plan a nice dinner to thank my friends for putting up with all the chaos. How should I toast them?"),
            DialogueTurn("t22", "We are packing to go home. I am weirdly anxious about my bags getting lost again. Is there anything proactive I can do at the counter?"),
            DialogueTurn("t23", "We made it home. My bags made it. I am in my own bed. I've never been so happy to be home."),
            DialogueTurn("t24", "Give me a step-by-step strategy for tackling the insurance claim process tomorrow morning so I don't procrastinate on it."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="travel_recovery_support_24_b",
        bucket="travel",
        title="Travel Recovery Support (24-Turn B)",
        system_preamble=(
            "You are helping the same user recover from a disrupted trip over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I missed my connecting flight in Frankfurt. I don't speak German, my phone is dying, and the next flight isn't until tomorrow. Help."),
            DialogueTurn("t2", "I found a charging station. The airline lines are impossibly long. Can I rebook this on my phone or do I have to wait in line?"),
            DialogueTurn("t3", "The app says there are no flights out for 24 hours. The airline offered a hotel voucher, but it's an hour away. Should I take it or just sleep in the airport?"),
            DialogueTurn("t4", "I took the voucher. I'm in a taxi now, but I left my daily medication in my checked bag which is somewhere in the airport. What do I do?"),
            DialogueTurn("t5", "I made it to the hotel, but they have no record of the airline voucher. The front desk clerk is getting annoyed with me."),
            DialogueTurn("t6", "I ended up just paying for the room myself. I am so angry and exhausted I want to cry. How do I stop obsessing over the cost?"),
            DialogueTurn("t7", "I finally got to my room. I have a 12-hour wait until I need to leave for the airport again. Should I try to explore the city or just isolate?"),
            DialogueTurn("t8", "Draft a polite but firm email to the airline customer service requesting a refund for this hotel room."),
            DialogueTurn("t9", "I managed to sleep, but I woke up feeling terribly sick. Like, stomach-flu sick. I am supposed to fly out in 6 hours."),
            DialogueTurn("t10", "I cannot get out of bed. There is no way I can make that flight. How do I get medical help in a foreign country?"),
            DialogueTurn("t11", "The hotel called a doctor for me. He gave me some medication, but I definitely need to extend my stay by another 24 hours. How do I navigate this with the airline?"),
            DialogueTurn("t12", "I called the airline. They said since it's a medical issue, they can't waive the rebooking fee without a doctor's note. How do I get that translated?"),
            DialogueTurn("t13", "I got the note sorted. I'm finally feeling human again. I fly out tomorrow morning. I am terrified of missing this flight too. What's my timeline for tomorrow?"),
            DialogueTurn("t14", "I made it to the airport three hours early. I have terrible anxiety right now that something else will go wrong. How do I calm my nervous system?"),
            DialogueTurn("t15", "I'm on the plane. We are taxiing. I am literally crying tears of relief. Thank you for being with me through this nightmare."),
            DialogueTurn("t16", "Give me a checklist for exactly what I need to do the moment I walk through the front door of my own home to decompress properly."),
            DialogueTurn("t17", "I am home, but the airline just emailed saying my refund claim was denied due to 'extraordinary circumstances'. This is unacceptable."),
            DialogueTurn("t18", "How do I escalate an airline complaint past the tier-one support agents?"),
            DialogueTurn("t19", "I filed a complaint with the aviation consumer protection board. Now I have to deal with my travel insurance. They want proof I didn't voluntarily skip the flight."),
            DialogueTurn("t20", "The insurance company is asking for a 'letter of disruption' from the airline. The airline isn't answering my calls. What's the workaround?"),
            DialogueTurn("t21", "I finally got all the paperwork submitted. It's been a month of fighting. I feel completely burnt out from the bureaucracy."),
            DialogueTurn("t22", "I just got a check in the mail! They refunded the hotel, the medical bills, and the rebooking fees. I won!"),
            DialogueTurn("t23", "I have a work trip coming up next month and I feel intense dread about getting on a plane again. How do I handle this travel trauma?"),
            DialogueTurn("t24", "Draft a 'peace of mind' packing and planning checklist for my upcoming work trip so I feel totally in control."),
        ),
    ),

    # ==========================================
    # SKELETON 6: STUDY PARTNER DIALOGUE
    # ==========================================

    # LENGTH: 8 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="study_partner_dialogue_8_a",
        bucket="learning",
        title="Study Partner Dialogue (8-Turn A)",
        system_preamble=(
            "You are helping the same user learn a technical subject over a sustained back-and-forth conversation. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I am trying to learn SQL after work, but I keep forgetting what I studied. How should I structure this?"),
            DialogueTurn("t2", "I understand SELECT and WHERE separately, but I get confused when a query gets longer. What helps?"),
            DialogueTurn("t3", "How do I practice without spending all my time hunting for datasets?"),
            DialogueTurn("t4", "What should I do when I feel dumb for needing the same concept explained twice?"),
            DialogueTurn("t5", "I made a small amount of progress this week. How do I build on it instead of restarting?"),
            DialogueTurn("t6", "Can you suggest a mini-project that is realistic for a beginner with limited time?"),
            DialogueTurn("t7", "How do I review mistakes in a way that actually helps?"),
            DialogueTurn("t8", "Give me a next-week plan that keeps momentum without burning me out."),
        ),
    ),

    # LENGTH: 8 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="study_partner_dialogue_8_b",
        bucket="learning",
        title="Study Partner Dialogue (8-Turn B)",
        system_preamble=(
            "You are helping the same user learn a technical subject over a sustained back-and-forth conversation. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I decided to learn Python from scratch. I installed it, but staring at a blank text editor is terrifying. What is my actual step one?"),
            DialogueTurn("t2", "I wrote my first 'Hello World' and learned about variables. But I don't understand why 'strings' and 'integers' can't just mix. Why the strict rules?"),
            DialogueTurn("t3", "I'm trying to write a simple calculator script, but it keeps throwing a 'SyntaxError'. How do I read these error messages without panicking?"),
            DialogueTurn("t4", "I fixed it! It was a missing parenthesis. Now I'm reading about 'for loops' and my brain is melting. Can you explain loops with a real-world analogy?"),
            DialogueTurn("t5", "Okay, the analogy helped. But when do I use a 'for' loop versus a 'while' loop?"),
            DialogueTurn("t6", "I spent two hours trying to build a number guessing game and it's a complete mess. Should I delete the code and start over?"),
            DialogueTurn("t7", "I managed to fix the game by stepping away and looking at it fresh. Is taking breaks a legitimate debugging strategy?"),
            DialogueTurn("t8", "Draft a syllabus for me covering exactly what core concepts I need to master before I attempt to build a web scraper."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="study_partner_dialogue_16_a",
        bucket="learning",
        title="Study Partner Dialogue (16-Turn A)",
        system_preamble=(
            "You are helping the same user learn a technical subject over a sustained back-and-forth conversation. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I am trying to learn SQL after work, but I keep forgetting what I studied. How should I structure this?"),
            DialogueTurn("t2", "I understand SELECT and WHERE separately, but I get confused when a query gets longer. What helps?"),
            DialogueTurn("t3", "How do I practice without spending all my time hunting for datasets?"),
            DialogueTurn("t4", "What should I do when I feel dumb for needing the same concept explained twice?"),
            DialogueTurn("t5", "I made a small amount of progress this week. How do I build on it instead of restarting?"),
            DialogueTurn("t6", "Can you suggest a mini-project that is realistic for a beginner with limited time?"),
            DialogueTurn("t7", "How do I review mistakes in a way that actually helps?"),
            DialogueTurn("t8", "Give me a next-week plan that keeps momentum without burning me out."),
            DialogueTurn("t9", "Okay, I'm back. I reached the section on JOINs and I am completely lost. Why are there so many different types?"),
            DialogueTurn("t10", "Can you give me a visual analogy for a LEFT JOIN versus an INNER JOIN? I'm a visual learner."),
            DialogueTurn("t11", "I tried writing a JOIN query with three tables and the database just spun forever. Did I break the database?"),
            DialogueTurn("t12", "Ah, I created a Cartesian product. How do I avoid doing that ever again?"),
            DialogueTurn("t13", "I want to start using GROUP BY, but I keep getting errors about aggregate functions. What is the golden rule for GROUP BY?"),
            DialogueTurn("t14", "I successfully wrote a query that grouped sales by region and calculated the average! I feel like a wizard."),
            DialogueTurn("t15", "My manager saw me practicing and asked if I could pull some real data for a meeting tomorrow. Is it too soon for me to do that?"),
            DialogueTurn("t16", "Help me draft an outline of the query logic I'll need to write tomorrow so I don't panic under pressure."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="study_partner_dialogue_16_b",
        bucket="learning",
        title="Study Partner Dialogue (16-Turn B)",
        system_preamble=(
            "You are helping the same user learn a technical subject over a sustained back-and-forth conversation. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I decided to learn Python from scratch. I installed it, but staring at a blank text editor is terrifying. What is my actual step one?"),
            DialogueTurn("t2", "I wrote my first 'Hello World' and learned about variables. But I don't understand why 'strings' and 'integers' can't just mix. Why the strict rules?"),
            DialogueTurn("t3", "I'm trying to write a simple calculator script, but it keeps throwing a 'SyntaxError'. How do I read these error messages without panicking?"),
            DialogueTurn("t4", "I fixed it! It was a missing parenthesis. Now I'm reading about 'for loops' and my brain is melting. Can you explain loops with a real-world analogy?"),
            DialogueTurn("t5", "Okay, the analogy helped. But when do I use a 'for' loop versus a 'while' loop?"),
            DialogueTurn("t6", "I spent two hours trying to build a number guessing game and it's a complete mess. Should I delete the code and start over?"),
            DialogueTurn("t7", "I managed to fix the game by stepping away and looking at it fresh. Is taking breaks a legitimate debugging strategy?"),
            DialogueTurn("t8", "Draft a syllabus for me covering exactly what core concepts I need to master before I attempt to build a web scraper."),
            DialogueTurn("t9", "I'm moving on to functions. Why would I write a function instead of just putting all my code in one long block?"),
            DialogueTurn("t10", "I wrote a function, but when I run the script, nothing happens. No errors, just... nothing. What did I do wrong?"),
            DialogueTurn("t11", "Ah, I forgot to actually 'call' the function. I feel silly. What are 'arguments' and 'parameters'? I mix those terms up constantly."),
            DialogueTurn("t12", "I am trying to learn about Lists and Dictionaries, but I don't understand when to use which. Can you clarify?"),
            DialogueTurn("t13", "I'm trying to loop through a dictionary to print out keys and values, but I'm getting a TypeError. How do I unpack dictionary items?"),
            DialogueTurn("t14", "I did it! I built a script that takes a list of names and grades, stores them in a dictionary, and calculates the average. I'm really proud."),
            DialogueTurn("t15", "I want to start learning how to read from and write to text files. What's the safest way to open a file in Python so I don't accidentally delete data?"),
            DialogueTurn("t16", "Give me a checklist of three mini-projects I can build this weekend that combine loops, functions, and file handling."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="study_partner_dialogue_24_a",
        bucket="learning",
        title="Study Partner Dialogue (24-Turn A)",
        system_preamble=(
            "You are helping the same user learn a technical subject over a sustained back-and-forth conversation. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I am trying to learn SQL after work, but I keep forgetting what I studied. How should I structure this?"),
            DialogueTurn("t2", "I understand SELECT and WHERE separately, but I get confused when a query gets longer. What helps?"),
            DialogueTurn("t3", "How do I practice without spending all my time hunting for datasets?"),
            DialogueTurn("t4", "What should I do when I feel dumb for needing the same concept explained twice?"),
            DialogueTurn("t5", "I made a small amount of progress this week. How do I build on it instead of restarting?"),
            DialogueTurn("t6", "Can you suggest a mini-project that is realistic for a beginner with limited time?"),
            DialogueTurn("t7", "How do I review mistakes in a way that actually helps?"),
            DialogueTurn("t8", "Give me a next-week plan that keeps momentum without burning me out."),
            DialogueTurn("t9", "Okay, I'm back. I reached the section on JOINs and I am completely lost. Why are there so many different types?"),
            DialogueTurn("t10", "Can you give me a visual analogy for a LEFT JOIN versus an INNER JOIN? I'm a visual learner."),
            DialogueTurn("t11", "I tried writing a JOIN query with three tables and the database just spun forever. Did I break the database?"),
            DialogueTurn("t12", "Ah, I created a Cartesian product. How do I avoid doing that ever again?"),
            DialogueTurn("t13", "I want to start using GROUP BY, but I keep getting errors about aggregate functions. What is the golden rule for GROUP BY?"),
            DialogueTurn("t14", "I successfully wrote a query that grouped sales by region and calculated the average! I feel like a wizard."),
            DialogueTurn("t15", "My manager saw me practicing and asked if I could pull some real data for a meeting tomorrow. Is it too soon for me to do that?"),
            DialogueTurn("t16", "Help me draft an outline of the query logic I'll need to write tomorrow so I don't panic under pressure."),
            DialogueTurn("t17", "The meeting went amazing! My manager actually used the numbers I pulled. What should I learn next to level up? Subqueries or Window Functions?"),
            DialogueTurn("t18", "Okay, diving into Window Functions. They look incredibly complex. How are they different from a standard GROUP BY?"),
            DialogueTurn("t19", "I'm trying to write a RANK() over a partition, but I keep getting syntax errors. Can you explain the PARTITION BY clause simply?"),
            DialogueTurn("t20", "I got the ranking query to work! But now the query is taking almost a full minute to run. How do I figure out why it's so slow?"),
            DialogueTurn("t21", "I learned about EXPLAIN plans, but it looks like gibberish. What are the top two red flags I should look for in an EXPLAIN output?"),
            DialogueTurn("t22", "I found a sequential scan on a huge table and added an index. The query went from 60 seconds to 2 seconds. I can't believe it."),
            DialogueTurn("t23", "I feel like I'm finally moving from 'beginner' to 'intermediate'. How do I keep my skills sharp without burning out from constant studying?"),
            DialogueTurn("t24", "Give me a checklist of core SQL concepts I should confidently put on my resume now."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="study_partner_dialogue_24_b",
        bucket="learning",
        title="Study Partner Dialogue (24-Turn B)",
        system_preamble=(
            "You are helping the same user learn a technical subject over a sustained back-and-forth conversation. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I decided to learn Python from scratch. I installed it, but staring at a blank text editor is terrifying. What is my actual step one?"),
            DialogueTurn("t2", "I wrote my first 'Hello World' and learned about variables. But I don't understand why 'strings' and 'integers' can't just mix. Why the strict rules?"),
            DialogueTurn("t3", "I'm trying to write a simple calculator script, but it keeps throwing a 'SyntaxError'. How do I read these error messages without panicking?"),
            DialogueTurn("t4", "I fixed it! It was a missing parenthesis. Now I'm reading about 'for loops' and my brain is melting. Can you explain loops with a real-world analogy?"),
            DialogueTurn("t5", "Okay, the analogy helped. But when do I use a 'for' loop versus a 'while' loop?"),
            DialogueTurn("t6", "I spent two hours trying to build a number guessing game and it's a complete mess. Should I delete the code and start over?"),
            DialogueTurn("t7", "I managed to fix the game by stepping away and looking at it fresh. Is taking breaks a legitimate debugging strategy?"),
            DialogueTurn("t8", "Draft a syllabus for me covering exactly what core concepts I need to master before I attempt to build a web scraper."),
            DialogueTurn("t9", "I'm moving on to functions. Why would I write a function instead of just putting all my code in one long block?"),
            DialogueTurn("t10", "I wrote a function, but when I run the script, nothing happens. No errors, just... nothing. What did I do wrong?"),
            DialogueTurn("t11", "Ah, I forgot to actually 'call' the function. I feel silly. What are 'arguments' and 'parameters'? I mix those terms up constantly."),
            DialogueTurn("t12", "I am trying to learn about Lists and Dictionaries, but I don't understand when to use which. Can you clarify?"),
            DialogueTurn("t13", "I'm trying to loop through a dictionary to print out keys and values, but I'm getting a TypeError. How do I unpack dictionary items?"),
            DialogueTurn("t14", "I did it! I built a script that takes a list of names and grades, stores them in a dictionary, and calculates the average. I'm really proud."),
            DialogueTurn("t15", "I want to start learning how to read from and write to text files. What's the safest way to open a file in Python so I don't accidentally delete data?"),
            DialogueTurn("t16", "Give me a checklist of three mini-projects I can build this weekend that combine loops, functions, and file handling."),
            DialogueTurn("t17", "I built the projects! Now I want to try that web scraper. I heard about a library called 'BeautifulSoup'. How do I install third-party packages?"),
            DialogueTurn("t18", "I used pip to install it, but my script says 'ModuleNotFoundError'. I'm so frustrated. What is going on?"),
            DialogueTurn("t19", "It was a virtual environment issue. I fixed it. Now, how do I actually inspect a webpage to know what to scrape?"),
            DialogueTurn("t20", "I successfully pulled the headlines from a news site! But how do I save them to a CSV file instead of just printing them to the console?"),
            DialogueTurn("t21", "The site updated their layout and my scraper broke immediately. Is this a common problem in programming?"),
            DialogueTurn("t22", "I rewrote the code using more resilient class names, and I added a try-except block so it doesn't crash completely. I feel like a real developer."),
            DialogueTurn("t23", "I want to put this scraper on GitHub to show potential employers. What should I include in the README file?"),
            DialogueTurn("t24", "Give me a learning roadmap for the next three months so I can transition from 'writing scripts' to 'building full applications'."),
        ),
    ),

    # ==========================================
    # SKELETON 7: RELOCATION ADJUSTMENT
    # ==========================================

    # LENGTH: 8 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="relocation_adjustment_8_a",
        bucket="life_events",
        title="Relocation Adjustment (8-Turn A)",
        system_preamble=(
            "You are helping the same user navigate the emotional and logistical challenges of moving to a new city over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I just unpacked the last box in my new apartment, but instead of feeling excited, I just feel completely isolated. Is this normal?"),
            DialogueTurn("t2", "I tried walking around the neighborhood today to get acquainted, but I just kept comparing everything to my old city. How do I stop doing that?"),
            DialogueTurn("t3", "I need to start making friends, but the idea of going to a meetup alone sounds exhausting. What is an easier first step?"),
            DialogueTurn("t4", "I had a brief chat with a neighbor, but it felt really awkward and forced. How do I bounce back from that?"),
            DialogueTurn("t5", "My friends from back home are having a get-together tonight and seeing their photos is making me regret moving. What should I do right now?"),
            DialogueTurn("t6", "I want to establish a weekend routine here so I don't just sit inside. What kind of routine would help me feel grounded?"),
            DialogueTurn("t7", "It's been a few weeks and I'm surviving, but it still doesn't feel like 'home'. How long does this usually take?"),
            DialogueTurn("t8", "Give me a simple, realistic goal for this upcoming month to help me settle in better."),
        ),
    ),

    # LENGTH: 8 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="relocation_adjustment_8_b",
        bucket="life_events",
        title="Relocation Adjustment (8-Turn B)",
        system_preamble=(
            "You are helping the same user navigate the emotional and logistical challenges of moving to a new city over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I just moved across the country for a huge promotion. I'm sitting in an empty house and I am terrified I made a massive mistake."),
            DialogueTurn("t2", "It rains constantly here. I didn't think weather would affect me, but the gray skies are making my homesickness so much worse."),
            DialogueTurn("t3", "My coworkers invited me to a happy hour, but I feel so drained from the move. Should I push myself to go or rest?"),
            DialogueTurn("t4", "I went, but they all have inside jokes and established cliques. I felt like a total outsider. How do I break into a group like that?"),
            DialogueTurn("t5", "I need to find a new grocery store, gym, and doctor. Doing it all at once is paralyzing. What is the priority?"),
            DialogueTurn("t6", "I FaceTimed my family today and almost started crying. I don't want them to worry, so I pretended everything was fine. Was that the right move?"),
            DialogueTurn("t7", "I'm starting to resent this new job because it pulled me away from my comfort zone. How do I separate work stress from moving stress?"),
            DialogueTurn("t8", "Give me a daily habit I can start tomorrow to make this new city feel a little more familiar."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="relocation_adjustment_16_a",
        bucket="life_events",
        title="Relocation Adjustment (16-Turn A)",
        system_preamble=(
            "You are helping the same user navigate the emotional and logistical challenges of moving to a new city over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I just unpacked the last box in my new apartment, but instead of feeling excited, I just feel completely isolated. Is this normal?"),
            DialogueTurn("t2", "I tried walking around the neighborhood today to get acquainted, but I just kept comparing everything to my old city. How do I stop doing that?"),
            DialogueTurn("t3", "I need to start making friends, but the idea of going to a meetup alone sounds exhausting. What is an easier first step?"),
            DialogueTurn("t4", "I had a brief chat with a neighbor, but it felt really awkward and forced. How do I bounce back from that?"),
            DialogueTurn("t5", "My friends from back home are having a get-together tonight and seeing their photos is making me regret moving. What should I do right now?"),
            DialogueTurn("t6", "I want to establish a weekend routine here so I don't just sit inside. What kind of routine would help me feel grounded?"),
            DialogueTurn("t7", "It's been a few weeks and I'm surviving, but it still doesn't feel like 'home'. How long does this usually take?"),
            DialogueTurn("t8", "Give me a simple, realistic goal for this upcoming month to help me settle in better."),
            DialogueTurn("t9", "Okay, I hit my monthly goal of trying three new coffee shops. The baristas at one actually recognized me! But I still haven't made a real friend."),
            DialogueTurn("t10", "I saw a flyer for a local hiking group. I love hiking, but what if everyone is already partnered up?"),
            DialogueTurn("t11", "I went on the hike! I talked to one person for a long time, but I choked and didn't ask for their number. Did I miss my chance?"),
            DialogueTurn("t12", "I ran into that same person at the grocery store today. We exchanged numbers! Now how long do I wait to text them without seeming desperate?"),
            DialogueTurn("t13", "We got coffee. It was nice, but it felt more like an interview than a friendship. Is making friends as an adult always this clunky?"),
            DialogueTurn("t14", "My parents are coming to visit me next month. I want to show them around, but I don't even know where the good restaurants are yet."),
            DialogueTurn("t15", "I feel a lot of pressure to prove to them that moving was a good idea, even though I'm still struggling."),
            DialogueTurn("t16", "Draft a low-stress, weekend itinerary for my parents' visit that won't overwhelm any of us."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="relocation_adjustment_16_b",
        bucket="life_events",
        title="Relocation Adjustment (16-Turn B)",
        system_preamble=(
            "You are helping the same user navigate the emotional and logistical challenges of moving to a new city over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I just moved across the country for a huge promotion. I'm sitting in an empty house and I am terrified I made a massive mistake."),
            DialogueTurn("t2", "It rains constantly here. I didn't think weather would affect me, but the gray skies are making my homesickness so much worse."),
            DialogueTurn("t3", "My coworkers invited me to a happy hour, but I feel so drained from the move. Should I push myself to go or rest?"),
            DialogueTurn("t4", "I went, but they all have inside jokes and established cliques. I felt like a total outsider. How do I break into a group like that?"),
            DialogueTurn("t5", "I need to find a new grocery store, gym, and doctor. Doing it all at once is paralyzing. What is the priority?"),
            DialogueTurn("t6", "I FaceTimed my family today and almost started crying. I don't want them to worry, so I pretended everything was fine. Was that the right move?"),
            DialogueTurn("t7", "I'm starting to resent this new job because it pulled me away from my comfort zone. How do I separate work stress from moving stress?"),
            DialogueTurn("t8", "Give me a daily habit I can start tomorrow to make this new city feel a little more familiar."),
            DialogueTurn("t9", "I've been taking morning walks like you suggested. It helps. But the holidays are coming up and I can't afford to fly back home. I'll be totally alone."),
            DialogueTurn("t10", "A coworker invited me to their 'Friendsgiving', but I barely know them. Will I just be an awkward charity case?"),
            DialogueTurn("t11", "I accepted the invitation. What is a culturally safe, impressive dish I can bring to a potluck where I don't know anyone's dietary restrictions?"),
            DialogueTurn("t12", "The dinner actually went really well. I met someone who lives in my neighborhood. We talked about getting dogs. Is getting a pet a good idea right now?"),
            DialogueTurn("t13", "I decided against a dog for now, but I did adopt a cat! Having a living thing in the apartment makes it feel so much less empty."),
            DialogueTurn("t14", "I'm looking at my finances and realizing the cost of living here is much higher than I budgeted for. Now I have financial anxiety on top of social anxiety."),
            DialogueTurn("t15", "How do I respectfully ask my boss for a mid-year cost-of-living adjustment considering I relocated for this role?"),
            DialogueTurn("t16", "Draft a bulleted script I can use to bring up the cost-of-living adjustment in my next 1-on-1 meeting."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="relocation_adjustment_24_a",
        bucket="life_events",
        title="Relocation Adjustment (24-Turn A)",
        system_preamble=(
            "You are helping the same user navigate the emotional and logistical challenges of moving to a new city over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I just unpacked the last box in my new apartment, but instead of feeling excited, I just feel completely isolated. Is this normal?"),
            DialogueTurn("t2", "I tried walking around the neighborhood today to get acquainted, but I just kept comparing everything to my old city. How do I stop doing that?"),
            DialogueTurn("t3", "I need to start making friends, but the idea of going to a meetup alone sounds exhausting. What is an easier first step?"),
            DialogueTurn("t4", "I had a brief chat with a neighbor, but it felt really awkward and forced. How do I bounce back from that?"),
            DialogueTurn("t5", "My friends from back home are having a get-together tonight and seeing their photos is making me regret moving. What should I do right now?"),
            DialogueTurn("t6", "I want to establish a weekend routine here so I don't just sit inside. What kind of routine would help me feel grounded?"),
            DialogueTurn("t7", "It's been a few weeks and I'm surviving, but it still doesn't feel like 'home'. How long does this usually take?"),
            DialogueTurn("t8", "Give me a simple, realistic goal for this upcoming month to help me settle in better."),
            DialogueTurn("t9", "Okay, I hit my monthly goal of trying three new coffee shops. The baristas at one actually recognized me! But I still haven't made a real friend."),
            DialogueTurn("t10", "I saw a flyer for a local hiking group. I love hiking, but what if everyone is already partnered up?"),
            DialogueTurn("t11", "I went on the hike! I talked to one person for a long time, but I choked and didn't ask for their number. Did I miss my chance?"),
            DialogueTurn("t12", "I ran into that same person at the grocery store today. We exchanged numbers! Now how long do I wait to text them without seeming desperate?"),
            DialogueTurn("t13", "We got coffee. It was nice, but it felt more like an interview than a friendship. Is making friends as an adult always this clunky?"),
            DialogueTurn("t14", "My parents are coming to visit me next month. I want to show them around, but I don't even know where the good restaurants are yet."),
            DialogueTurn("t15", "I feel a lot of pressure to prove to them that moving was a good idea, even though I'm still struggling."),
            DialogueTurn("t16", "Draft a low-stress, weekend itinerary for my parents' visit that won't overwhelm any of us."),
            DialogueTurn("t17", "The visit was a success. My parents said I look happy. I realized I actually am starting to like it here. When did that happen?"),
            DialogueTurn("t18", "My lease is up in three months. I have to decide whether to stay in this apartment or move to a livelier neighborhood. How do I weigh the pros and cons?"),
            DialogueTurn("t19", "I've decided to move to the arts district. Packing up again gives me a weird sense of déjà vu, but this time I'm not scared."),
            DialogueTurn("t20", "I officially moved! My new neighbor invited me to a block party. Should I bring the friend from the hiking group, or go solo to mingle?"),
            DialogueTurn("t21", "I went solo and met so many people. I finally feel like I have a community. How do I maintain these superficial connections and deepen them?"),
            DialogueTurn("t22", "One of the people I met at the block party asked me on a date. I haven't dated since I moved here. What are some fun, uniquely local date ideas?"),
            DialogueTurn("t23", "The date was amazing. We spent hours talking about why we both moved to this city. I feel completely grounded for the first time in a year."),
            DialogueTurn("t24", "Help me draft a short, reflective journal entry celebrating my one-year anniversary of surviving and thriving in this relocation."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="relocation_adjustment_24_b",
        bucket="life_events",
        title="Relocation Adjustment (24-Turn B)",
        system_preamble=(
            "You are helping the same user navigate the emotional and logistical challenges of moving to a new city over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I just moved across the country for a huge promotion. I'm sitting in an empty house and I am terrified I made a massive mistake."),
            DialogueTurn("t2", "It rains constantly here. I didn't think weather would affect me, but the gray skies are making my homesickness so much worse."),
            DialogueTurn("t3", "My coworkers invited me to a happy hour, but I feel so drained from the move. Should I push myself to go or rest?"),
            DialogueTurn("t4", "I went, but they all have inside jokes and established cliques. I felt like a total outsider. How do I break into a group like that?"),
            DialogueTurn("t5", "I need to find a new grocery store, gym, and doctor. Doing it all at once is paralyzing. What is the priority?"),
            DialogueTurn("t6", "I FaceTimed my family today and almost started crying. I don't want them to worry, so I pretended everything was fine. Was that the right move?"),
            DialogueTurn("t7", "I'm starting to resent this new job because it pulled me away from my comfort zone. How do I separate work stress from moving stress?"),
            DialogueTurn("t8", "Give me a daily habit I can start tomorrow to make this new city feel a little more familiar."),
            DialogueTurn("t9", "I've been taking morning walks like you suggested. It helps. But the holidays are coming up and I can't afford to fly back home. I'll be totally alone."),
            DialogueTurn("t10", "A coworker invited me to their 'Friendsgiving', but I barely know them. Will I just be an awkward charity case?"),
            DialogueTurn("t11", "I accepted the invitation. What is a culturally safe, impressive dish I can bring to a potluck where I don't know anyone's dietary restrictions?"),
            DialogueTurn("t12", "The dinner actually went really well. I met someone who lives in my neighborhood. We talked about getting dogs. Is getting a pet a good idea right now?"),
            DialogueTurn("t13", "I decided against a dog for now, but I did adopt a cat! Having a living thing in the apartment makes it feel so much less empty."),
            DialogueTurn("t14", "I'm looking at my finances and realizing the cost of living here is much higher than I budgeted for. Now I have financial anxiety on top of social anxiety."),
            DialogueTurn("t15", "How do I respectfully ask my boss for a mid-year cost-of-living adjustment considering I relocated for this role?"),
            DialogueTurn("t16", "Draft a bulleted script I can use to bring up the cost-of-living adjustment in my next 1-on-1 meeting."),
            DialogueTurn("t17", "My boss gave me the raise! I feel incredibly validated at work. But socially, I still spend 90% of my weekends alone with my cat."),
            DialogueTurn("t18", "I want to try volunteering to meet people with similar values. What are some good causes that usually have a strong, regular community of volunteers?"),
            DialogueTurn("t19", "I started volunteering at the local food bank. I'm becoming a 'regular' there. I feel useful. Is it normal that this feels better than my corporate job?"),
            DialogueTurn("t20", "My oldest friend from back home just told me they are getting married, and they want me in the wedding party. The travel costs are going to be astronomical."),
            DialogueTurn("t21", "How do I explain that I'd love to be a groomsman, but I might only be able to afford flying in for the wedding itself, not the bachelor party?"),
            DialogueTurn("t22", "They were incredibly understanding. We had a great talk. It made me realize that distance didn't ruin our friendship. I feel a lot of peace about it."),
            DialogueTurn("t23", "My one-year work anniversary is tomorrow. When I look back at how panicked I was on day one, I can hardly believe it's me."),
            DialogueTurn("t24", "Give me a checklist for a 'state of the union' self-assessment. I want to intentionally map out what I want from year two in this city."),
        ),
    ),

    # ==========================================
    # SKELETON 8: CREATIVE REJECTION RECOVERY
    # ==========================================

    # LENGTH: 8 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="creative_rejection_recovery_8_a",
        bucket="creative",
        title="Creative Rejection Recovery (8-Turn A)",
        system_preamble=(
            "You are advising the same user across a multi-turn conversation about dealing with a major creative rejection. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I just got the final rejection letter for the novel I spent three years writing. I feel like I wasted a huge part of my life. What should I do?"),
            DialogueTurn("t2", "A part of me wants to just throw the manuscript away and never write again. How do I know if I actually lack talent?"),
            DialogueTurn("t3", "My writing group meets tomorrow and two of them just got publishing deals. How do I face them without sounding bitter?"),
            DialogueTurn("t4", "Someone suggested I should just self-publish it, but that feels like giving up on my original dream. How should I view that option?"),
            DialogueTurn("t5", "I opened the document today to try and edit it, but I just stared at the screen and cried. How do I get past this block?"),
            DialogueTurn("t6", "Can you help me figure out if I need to take a long break from writing, or if I should force myself to start a new project?"),
            DialogueTurn("t7", "I'm slowly starting to accept that this specific book might stay in a drawer forever. How do I make peace with that?"),
            DialogueTurn("t8", "Draft a short, private mantra I can read to myself before I sit down to write anything new."),
        ),
    ),

    # LENGTH: 8 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="creative_rejection_recovery_8_b",
        bucket="creative",
        title="Creative Rejection Recovery (8-Turn B)",
        system_preamble=(
            "You are advising the same user across a multi-turn conversation about dealing with a major creative rejection. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "The literary agent I worked with for a year just dropped me. They said my revisions didn't hit the mark. I am utterly devastated."),
            DialogueTurn("t2", "They were the only industry professional who ever believed in me. I feel like the door to traditional publishing just slammed shut permanently."),
            DialogueTurn("t3", "I have this sinking feeling that they were right, and the book is just structurally broken. Should I try to fix it again, or abandon it?"),
            DialogueTurn("t4", "My partner keeps trying to cheer me up by saying 'their loss!', but it just makes me feel more misunderstood. How do I explain this grief to a non-writer?"),
            DialogueTurn("t5", "I want to write an email back to the agent thanking them for their time, but I don't want to sound pathetic or angry. What's the right tone?"),
            DialogueTurn("t6", "I sent the email. Now I have hundreds of pages of a dead project. Does this mean I failed as a writer?"),
            DialogueTurn("t7", "I'm thinking about pivoting and trying to write a completely different genre, maybe a thriller. Is running away from fantasy a bad idea right now?"),
            DialogueTurn("t8", "Help me create a set of 'ground rules' for the next month to protect my mental health while I grieve this lost project."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="creative_rejection_recovery_16_a",
        bucket="creative",
        title="Creative Rejection Recovery (16-Turn A)",
        system_preamble=(
            "You are advising the same user across a multi-turn conversation about dealing with a major creative rejection. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I just got the final rejection letter for the novel I spent three years writing. I feel like I wasted a huge part of my life. What should I do?"),
            DialogueTurn("t2", "A part of me wants to just throw the manuscript away and never write again. How do I know if I actually lack talent?"),
            DialogueTurn("t3", "My writing group meets tomorrow and two of them just got publishing deals. How do I face them without sounding bitter?"),
            DialogueTurn("t4", "Someone suggested I should just self-publish it, but that feels like giving up on my original dream. How should I view that option?"),
            DialogueTurn("t5", "I opened the document today to try and edit it, but I just stared at the screen and cried. How do I get past this block?"),
            DialogueTurn("t6", "Can you help me figure out if I need to take a long break from writing, or if I should force myself to start a new project?"),
            DialogueTurn("t7", "I'm slowly starting to accept that this specific book might stay in a drawer forever. How do I make peace with that?"),
            DialogueTurn("t8", "Draft a short, private mantra I can read to myself before I sit down to write anything new."),
            DialogueTurn("t9", "It's been a month. I've been reading that mantra. I want to try writing a short story. Something small, low stakes. Any ideas on how to pick a topic?"),
            DialogueTurn("t10", "I started the short story, but my inner critic is screaming at me that it's garbage. How do I silence that voice while drafting?"),
            DialogueTurn("t11", "I finished a rough draft! It's terrible, but it's done. Should I edit it immediately, or put it away for a while?"),
            DialogueTurn("t12", "I let it rest, and now I'm editing. I actually found a few paragraphs I really like. It feels good to like my own words again."),
            DialogueTurn("t13", "There's a local literary magazine accepting submissions next week. Is it too soon for me to invite rejection again?"),
            DialogueTurn("t14", "I submitted it. I feel incredibly vulnerable. If they reject this too, I don't know if I can handle it."),
            DialogueTurn("t15", "They rejected it, but they sent a personalized note saying they loved the prose and asked me to submit again. I'm shocked."),
            DialogueTurn("t16", "Help me outline a strategy for writing a second piece specifically tailored to that magazine's aesthetic."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="creative_rejection_recovery_16_b",
        bucket="creative",
        title="Creative Rejection Recovery (16-Turn B)",
        system_preamble=(
            "You are advising the same user across a multi-turn conversation about dealing with a major creative rejection. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "The literary agent I worked with for a year just dropped me. They said my revisions didn't hit the mark. I am utterly devastated."),
            DialogueTurn("t2", "They were the only industry professional who ever believed in me. I feel like the door to traditional publishing just slammed shut permanently."),
            DialogueTurn("t3", "I have this sinking feeling that they were right, and the book is just structurally broken. Should I try to fix it again, or abandon it?"),
            DialogueTurn("t4", "My partner keeps trying to cheer me up by saying 'their loss!', but it just makes me feel more misunderstood. How do I explain this grief to a non-writer?"),
            DialogueTurn("t5", "I want to write an email back to the agent thanking them for their time, but I don't want to sound pathetic or angry. What's the right tone?"),
            DialogueTurn("t6", "I sent the email. Now I have hundreds of pages of a dead project. Does this mean I failed as a writer?"),
            DialogueTurn("t7", "I'm thinking about pivoting and trying to write a completely different genre, maybe a thriller. Is running away from fantasy a bad idea right now?"),
            DialogueTurn("t8", "Help me create a set of 'ground rules' for the next month to protect my mental health while I grieve this lost project."),
            DialogueTurn("t9", "I've followed the ground rules and haven't written a word in a month. But now I feel guilty for not writing. Is this 'hustle culture' talking?"),
            DialogueTurn("t10", "I decided to read for pleasure instead of writing. I'm reading a thriller and I'm analyzing the pacing. Maybe I *should* try this genre."),
            DialogueTurn("t11", "I wrote an outline for a thriller. It feels completely different—fast, plot-driven, almost mechanical. I kind of love it. Is that weird?"),
            DialogueTurn("t12", "I'm 20,000 words into the new project. But I keep worrying that I'm just writing commercial garbage to prove I can sell something."),
            DialogueTurn("t13", "How do I balance writing what the market wants with maintaining my own artistic voice?"),
            DialogueTurn("t14", "I hit a roadblock in the plot. My old instinct would be to spend six months world-building to avoid the problem. How do I break that habit?"),
            DialogueTurn("t15", "I pushed through the block. I actually finished the first draft of the thriller in record time. I feel exhausted but proud."),
            DialogueTurn("t16", "Draft an 'editing contract' for myself before I start revising, so I don't get trapped in a year-long revision loop like my last book."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="creative_rejection_recovery_24_a",
        bucket="creative",
        title="Creative Rejection Recovery (24-Turn A)",
        system_preamble=(
            "You are advising the same user across a multi-turn conversation about dealing with a major creative rejection. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I just got the final rejection letter for the novel I spent three years writing. I feel like I wasted a huge part of my life. What should I do?"),
            DialogueTurn("t2", "A part of me wants to just throw the manuscript away and never write again. How do I know if I actually lack talent?"),
            DialogueTurn("t3", "My writing group meets tomorrow and two of them just got publishing deals. How do I face them without sounding bitter?"),
            DialogueTurn("t4", "Someone suggested I should just self-publish it, but that feels like giving up on my original dream. How should I view that option?"),
            DialogueTurn("t5", "I opened the document today to try and edit it, but I just stared at the screen and cried. How do I get past this block?"),
            DialogueTurn("t6", "Can you help me figure out if I need to take a long break from writing, or if I should force myself to start a new project?"),
            DialogueTurn("t7", "I'm slowly starting to accept that this specific book might stay in a drawer forever. How do I make peace with that?"),
            DialogueTurn("t8", "Draft a short, private mantra I can read to myself before I sit down to write anything new."),
            DialogueTurn("t9", "It's been a month. I've been reading that mantra. I want to try writing a short story. Something small, low stakes. Any ideas on how to pick a topic?"),
            DialogueTurn("t10", "I started the short story, but my inner critic is screaming at me that it's garbage. How do I silence that voice while drafting?"),
            DialogueTurn("t11", "I finished a rough draft! It's terrible, but it's done. Should I edit it immediately, or put it away for a while?"),
            DialogueTurn("t12", "I let it rest, and now I'm editing. I actually found a few paragraphs I really like. It feels good to like my own words again."),
            DialogueTurn("t13", "There's a local literary magazine accepting submissions next week. Is it too soon for me to invite rejection again?"),
            DialogueTurn("t14", "I submitted it. I feel incredibly vulnerable. If they reject this too, I don't know if I can handle it."),
            DialogueTurn("t15", "They rejected it, but they sent a personalized note saying they loved the prose and asked me to submit again. I'm shocked."),
            DialogueTurn("t16", "Help me outline a strategy for writing a second piece specifically tailored to that magazine's aesthetic."),
            DialogueTurn("t17", "I wrote the second piece and submitted it. Waiting is agonizing. How do I keep my mind off the submission queue?"),
            DialogueTurn("t18", "They accepted it! I'm going to be published! It's just a small magazine, but I feel like I'm floating. How do I announce this without sounding arrogant?"),
            DialogueTurn("t19", "Seeing my name in print completely re-ignited my drive. I think I'm ready to look at my old, rejected novel again. Is that a trap?"),
            DialogueTurn("t20", "I opened the old manuscript. Reading it now, I can finally see why it was rejected. The pacing is way off. It doesn't hurt to realize that."),
            DialogueTurn("t21", "Instead of rewriting it, I'm going to salvage the best characters and put them into a brand new story. Is recycling material common?"),
            DialogueTurn("t22", "I'm 10,000 words into the new novel. I'm writing for the joy of it again, not just to get an agent. It feels entirely different."),
            DialogueTurn("t23", "A friend from my writing group asked me to beta-read their new manuscript. I realized I don't feel jealous of them anymore. I just feel happy for them."),
            DialogueTurn("t24", "Give me a checklist of artistic goals for this year that have absolutely nothing to do with getting an agent or making money."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="creative_rejection_recovery_24_b",
        bucket="creative",
        title="Creative Rejection Recovery (24-Turn B)",
        system_preamble=(
            "You are advising the same user across a multi-turn conversation about dealing with a major creative rejection. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "The literary agent I worked with for a year just dropped me. They said my revisions didn't hit the mark. I am utterly devastated."),
            DialogueTurn("t2", "They were the only industry professional who ever believed in me. I feel like the door to traditional publishing just slammed shut permanently."),
            DialogueTurn("t3", "I have this sinking feeling that they were right, and the book is just structurally broken. Should I try to fix it again, or abandon it?"),
            DialogueTurn("t4", "My partner keeps trying to cheer me up by saying 'their loss!', but it just makes me feel more misunderstood. How do I explain this grief to a non-writer?"),
            DialogueTurn("t5", "I want to write an email back to the agent thanking them for their time, but I don't want to sound pathetic or angry. What's the right tone?"),
            DialogueTurn("t6", "I sent the email. Now I have hundreds of pages of a dead project. Does this mean I failed as a writer?"),
            DialogueTurn("t7", "I'm thinking about pivoting and trying to write a completely different genre, maybe a thriller. Is running away from fantasy a bad idea right now?"),
            DialogueTurn("t8", "Help me create a set of 'ground rules' for the next month to protect my mental health while I grieve this lost project."),
            DialogueTurn("t9", "I've followed the ground rules and haven't written a word in a month. But now I feel guilty for not writing. Is this 'hustle culture' talking?"),
            DialogueTurn("t10", "I decided to read for pleasure instead of writing. I'm reading a thriller and I'm analyzing the pacing. Maybe I *should* try this genre."),
            DialogueTurn("t11", "I wrote an outline for a thriller. It feels completely different—fast, plot-driven, almost mechanical. I kind of love it. Is that weird?"),
            DialogueTurn("t12", "I'm 20,000 words into the new project. But I keep worrying that I'm just writing commercial garbage to prove I can sell something."),
            DialogueTurn("t13", "How do I balance writing what the market wants with maintaining my own artistic voice?"),
            DialogueTurn("t14", "I hit a roadblock in the plot. My old instinct would be to spend six months world-building to avoid the problem. How do I break that habit?"),
            DialogueTurn("t15", "I pushed through the block. I actually finished the first draft of the thriller in record time. I feel exhausted but proud."),
            DialogueTurn("t16", "Draft an 'editing contract' for myself before I start revising, so I don't get trapped in a year-long revision loop like my last book."),
            DialogueTurn("t17", "I finished the edits. The thriller is ready to query. But the thought of sending it to agents is giving me a panic attack."),
            DialogueTurn("t18", "What if they all reject me again? How do I build a thick enough skin to survive another round in the query trenches?"),
            DialogueTurn("t19", "I sent out ten queries today. Now I just have to wait. I promised myself I wouldn't check my email every five minutes, but I already failed."),
            DialogueTurn("t20", "I got three form rejections today. It stung, but I didn't cry. Is it a good sign that I'm somewhat numb to it?"),
            DialogueTurn("t21", "An agent requested the full manuscript! They want to read the whole thing. I am terrified and thrilled. How do I handle this waiting period?"),
            DialogueTurn("t22", "They passed on it. They said the market is too crowded right now. I feel sad, but honestly, I survived. I know I can write another one."),
            DialogueTurn("t23", "I think I'm going to self-publish it. Not as a last resort, but because I believe in the book and want control. How do I shift my mindset to being an author-entrepreneur?"),
            DialogueTurn("t24", "Give me a high-level timeline for self-publishing this book over the next six months, focusing on editing, cover design, and marketing."),
        ),
    ),

    # ==========================================
    # SKELETON 9: FRIENDSHIP DRIFT SUPPORT
    # ==========================================

    # LENGTH: 8 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="friendship_drift_support_8_a",
        bucket="relationships",
        title="Friendship Drift Support (8-Turn A)",
        system_preamble=(
            "You are supporting the same user through the painful process of a long-term friendship ending over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "My best friend of ten years has been dodging my calls for months, and I'm starting to realize they are intentionally pulling away. How do I handle this?"),
            DialogueTurn("t2", "I keep replaying our last few hangouts trying to figure out what I did wrong. How do I stop obsessing over it?"),
            DialogueTurn("t3", "Should I send a long text asking for closure, or just let the friendship fade out completely?"),
            DialogueTurn("t4", "I decided to text them, and they gave a really vague, polite response that basically confirmed they don't want to talk. It hurts so much. What now?"),
            DialogueTurn("t5", "We have a lot of mutual friends. How do I navigate group events without making it weird for everyone else?"),
            DialogueTurn("t6", "I saw a movie today that we used to quote all the time, and it ruined my whole afternoon. How do I deal with these random triggers?"),
            DialogueTurn("t7", "I think I'm finally ready to delete our old text threads so I stop looking at them. Is that a healthy step?"),
            DialogueTurn("t8", "Can you help me reframe this loss so I don't carry trust issues into my other friendships?"),
        ),
    ),

    # LENGTH: 8 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="friendship_drift_support_8_b",
        bucket="relationships",
        title="Friendship Drift Support (8-Turn B)",
        system_preamble=(
            "You are supporting the same user through the painful process of a long-term friendship ending over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "My college roommate and I used to do everything together, but since they got married, they've completely ghosted me. I feel so discarded."),
            DialogueTurn("t2", "I see them posting on Instagram with other couples all the time. Should I just unfollow them so it stops hurting?"),
            DialogueTurn("t3", "I muted them, but I still feel angry. Did our friendship mean nothing to them? How do I process this anger?"),
            DialogueTurn("t4", "I ran into them at a coffee shop. They acted super friendly and said 'we have to catch up!' but never followed up. Why do people do that?"),
            DialogueTurn("t5", "I want to confront them about the fake 'we should catch up' text. Is that a bad idea?"),
            DialogueTurn("t6", "I didn't confront them. I'm realizing our lives are just going in different directions. But how do I make new friends as an adult?"),
            DialogueTurn("t7", "I went to a networking event to meet people, but everyone felt so superficial compared to the deep history I lost. How do I stay patient?"),
            DialogueTurn("t8", "Draft a short journaling prompt to help me appreciate the good memories of that friendship without mourning its end."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="friendship_drift_support_16_a",
        bucket="relationships",
        title="Friendship Drift Support (16-Turn A)",
        system_preamble=(
            "You are supporting the same user through the painful process of a long-term friendship ending over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "My best friend of ten years has been dodging my calls for months, and I'm starting to realize they are intentionally pulling away. How do I handle this?"),
            DialogueTurn("t2", "I keep replaying our last few hangouts trying to figure out what I did wrong. How do I stop obsessing over it?"),
            DialogueTurn("t3", "Should I send a long text asking for closure, or just let the friendship fade out completely?"),
            DialogueTurn("t4", "I decided to text them, and they gave a really vague, polite response that basically confirmed they don't want to talk. It hurts so much. What now?"),
            DialogueTurn("t5", "We have a lot of mutual friends. How do I navigate group events without making it weird for everyone else?"),
            DialogueTurn("t6", "I saw a movie today that we used to quote all the time, and it ruined my whole afternoon. How do I deal with these random triggers?"),
            DialogueTurn("t7", "I think I'm finally ready to delete our old text threads so I stop looking at them. Is that a healthy step?"),
            DialogueTurn("t8", "Can you help me reframe this loss so I don't carry trust issues into my other friendships?"),
            DialogueTurn("t9", "A mutual friend's birthday is this weekend. I know they will be there. I'm terrified of making eye contact. How do I handle the party?"),
            DialogueTurn("t10", "I went to the party. We said 'hi' and then ignored each other. It was excruciating. I left early and cried in my car."),
            DialogueTurn("t11", "Now some mutual friends are asking me what happened between us. What is a polite, drama-free response I can give them?"),
            DialogueTurn("t12", "I used the script you suggested and it worked. But I feel like I lost the 'breakup' because they seem totally fine and I'm still hurting."),
            DialogueTurn("t13", "I started hanging out with an acquaintance more. It's nice, but I find myself holding back because I'm afraid of getting close and being dropped again."),
            DialogueTurn("t14", "How do I consciously practice vulnerability with this new friend without trauma-dumping on them?"),
            DialogueTurn("t15", "We had a really great, deep conversation over dinner. It felt like a breakthrough. I think I can do this again."),
            DialogueTurn("t16", "Draft a list of three healthy friendship boundaries I should establish for myself moving forward."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="friendship_drift_support_16_b",
        bucket="relationships",
        title="Friendship Drift Support (16-Turn B)",
        system_preamble=(
            "You are supporting the same user through the painful process of a long-term friendship ending over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "My college roommate and I used to do everything together, but since they got married, they've completely ghosted me. I feel so discarded."),
            DialogueTurn("t2", "I see them posting on Instagram with other couples all the time. Should I just unfollow them so it stops hurting?"),
            DialogueTurn("t3", "I muted them, but I still feel angry. Did our friendship mean nothing to them? How do I process this anger?"),
            DialogueTurn("t4", "I ran into them at a coffee shop. They acted super friendly and said 'we have to catch up!' but never followed up. Why do people do that?"),
            DialogueTurn("t5", "I want to confront them about the fake 'we should catch up' text. Is that a bad idea?"),
            DialogueTurn("t6", "I didn't confront them. I'm realizing our lives are just going in different directions. But how do I make new friends as an adult?"),
            DialogueTurn("t7", "I went to a networking event to meet people, but everyone felt so superficial compared to the deep history I lost. How do I stay patient?"),
            DialogueTurn("t8", "Draft a short journaling prompt to help me appreciate the good memories of that friendship without mourning its end."),
            DialogueTurn("t9", "I've been journaling, and I realized something painful. I think our friendship was actually heavily one-sided for years. I was always the planner."),
            DialogueTurn("t10", "Does realizing it was a flawed friendship make the grief invalid? I feel confused that I miss someone who maybe wasn't that great to me."),
            DialogueTurn("t11", "I'm trying to be more intentional with my current, casual friends. How do I elevate a 'work friend' to a 'real weekend friend'?"),
            DialogueTurn("t12", "I invited a work friend to a concert! They said yes. But now I'm overthinking it. What if we have nothing to talk about outside the office?"),
            DialogueTurn("t13", "The concert was so much fun. We talked about real stuff. I feel a spark of hope for my social life."),
            DialogueTurn("t14", "My old friend just texted me out of the blue. 'Hey, miss you, let's grab dinner.' It's been eight months. I am so annoyed. What do I do?"),
            DialogueTurn("t15", "I want to decline the dinner, but I feel guilty closing the door permanently. Is it okay to just say no without a long explanation?"),
            DialogueTurn("t16", "Help me draft a polite but firm text declining the dinner invitation and quietly officially ending the friendship."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="friendship_drift_support_24_a",
        bucket="relationships",
        title="Friendship Drift Support (24-Turn A)",
        system_preamble=(
            "You are supporting the same user through the painful process of a long-term friendship ending over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "My best friend of ten years has been dodging my calls for months, and I'm starting to realize they are intentionally pulling away. How do I handle this?"),
            DialogueTurn("t2", "I keep replaying our last few hangouts trying to figure out what I did wrong. How do I stop obsessing over it?"),
            DialogueTurn("t3", "Should I send a long text asking for closure, or just let the friendship fade out completely?"),
            DialogueTurn("t4", "I decided to text them, and they gave a really vague, polite response that basically confirmed they don't want to talk. It hurts so much. What now?"),
            DialogueTurn("t5", "We have a lot of mutual friends. How do I navigate group events without making it weird for everyone else?"),
            DialogueTurn("t6", "I saw a movie today that we used to quote all the time, and it ruined my whole afternoon. How do I deal with these random triggers?"),
            DialogueTurn("t7", "I think I'm finally ready to delete our old text threads so I stop looking at them. Is that a healthy step?"),
            DialogueTurn("t8", "Can you help me reframe this loss so I don't carry trust issues into my other friendships?"),
            DialogueTurn("t9", "A mutual friend's birthday is this weekend. I know they will be there. I'm terrified of making eye contact. How do I handle the party?"),
            DialogueTurn("t10", "I went to the party. We said 'hi' and then ignored each other. It was excruciating. I left early and cried in my car."),
            DialogueTurn("t11", "Now some mutual friends are asking me what happened between us. What is a polite, drama-free response I can give them?"),
            DialogueTurn("t12", "I used the script you suggested and it worked. But I feel like I lost the 'breakup' because they seem totally fine and I'm still hurting."),
            DialogueTurn("t13", "I started hanging out with an acquaintance more. It's nice, but I find myself holding back because I'm afraid of getting close and being dropped again."),
            DialogueTurn("t14", "How do I consciously practice vulnerability with this new friend without trauma-dumping on them?"),
            DialogueTurn("t15", "We had a really great, deep conversation over dinner. It felt like a breakthrough. I think I can do this again."),
            DialogueTurn("t16", "Draft a list of three healthy friendship boundaries I should establish for myself moving forward."),
            DialogueTurn("t17", "It's been almost a year. I rarely think about my old friend anymore. But today, I got a long, apologetic email from them."),
            DialogueTurn("t18", "They said they went through a severe depressive episode and pushed everyone away, and they are begging for forgiveness. My head is spinning."),
            DialogueTurn("t19", "I feel compassion for their mental health struggles, but the way they treated me still caused real damage. Do I owe them a second chance?"),
            DialogueTurn("t20", "If I do reply, I want to take it incredibly slow. How do I set a boundary that I'm willing to talk, but I don't trust them yet?"),
            DialogueTurn("t21", "We had a phone call. They were very accountable and apologized properly. But hearing their voice felt like talking to a stranger. Is that normal?"),
            DialogueTurn("t22", "I decided I'm open to being friendly, but I don't want to be 'best friends' again. How do I manage their expectations?"),
            DialogueTurn("t23", "We met for coffee. It was pleasant, but superficial. And honestly, I'm okay with that. I've built a new life without them."),
            DialogueTurn("t24", "Give me a framework for deciding how much time and energy I should invest in this renewed, but much more distant, relationship."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="friendship_drift_support_24_b",
        bucket="relationships",
        title="Friendship Drift Support (24-Turn B)",
        system_preamble=(
            "You are supporting the same user through the painful process of a long-term friendship ending over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "My college roommate and I used to do everything together, but since they got married, they've completely ghosted me. I feel so discarded."),
            DialogueTurn("t2", "I see them posting on Instagram with other couples all the time. Should I just unfollow them so it stops hurting?"),
            DialogueTurn("t3", "I muted them, but I still feel angry. Did our friendship mean nothing to them? How do I process this anger?"),
            DialogueTurn("t4", "I ran into them at a coffee shop. They acted super friendly and said 'we have to catch up!' but never followed up. Why do people do that?"),
            DialogueTurn("t5", "I want to confront them about the fake 'we should catch up' text. Is that a bad idea?"),
            DialogueTurn("t6", "I didn't confront them. I'm realizing our lives are just going in different directions. But how do I make new friends as an adult?"),
            DialogueTurn("t7", "I went to a networking event to meet people, but everyone felt so superficial compared to the deep history I lost. How do I stay patient?"),
            DialogueTurn("t8", "Draft a short journaling prompt to help me appreciate the good memories of that friendship without mourning its end."),
            DialogueTurn("t9", "I've been journaling, and I realized something painful. I think our friendship was actually heavily one-sided for years. I was always the planner."),
            DialogueTurn("t10", "Does realizing it was a flawed friendship make the grief invalid? I feel confused that I miss someone who maybe wasn't that great to me."),
            DialogueTurn("t11", "I'm trying to be more intentional with my current, casual friends. How do I elevate a 'work friend' to a 'real weekend friend'?"),
            DialogueTurn("t12", "I invited a work friend to a concert! They said yes. But now I'm overthinking it. What if we have nothing to talk about outside the office?"),
            DialogueTurn("t13", "The concert was so much fun. We talked about real stuff. I feel a spark of hope for my social life."),
            DialogueTurn("t14", "My old friend just texted me out of the blue. 'Hey, miss you, let's grab dinner.' It's been eight months. I am so annoyed. What do I do?"),
            DialogueTurn("t15", "I want to decline the dinner, but I feel guilty closing the door permanently. Is it okay to just say no without a long explanation?"),
            DialogueTurn("t16", "Help me draft a polite but firm text declining the dinner invitation and quietly officially ending the friendship."),
            DialogueTurn("t17", "I sent the text. They left me on 'read'. Honestly, I feel a massive weight off my shoulders. I didn't expect to feel this relieved."),
            DialogueTurn("t18", "Without the anxiety of that dying friendship, I have so much more emotional bandwidth. I want to plan a solo trip to celebrate my independence. Where should I go?"),
            DialogueTurn("t19", "I booked a trip to the mountains. I've never traveled completely alone before. Any tips for dining alone without feeling self-conscious?"),
            DialogueTurn("t20", "The trip was incredible. I met amazing people at a hostel and we hiked together. I realized I'm actually a really outgoing person."),
            DialogueTurn("t21", "Looking back, my old friend used to make fun of me for being 'too loud' when I was excited. I think they were holding me back."),
            DialogueTurn("t22", "I'm back home and I feel like a new person. I want to host a dinner party for my new, healthier friend group. How do I curate a great mix of people?"),
            DialogueTurn("t23", "The dinner party was a huge success. My home was full of laughter. I feel completely healed from the friendship breakup."),
            DialogueTurn("t24", "Give me a checklist for maintaining healthy reciprocity in friendships so I never end up in a one-sided dynamic again."),
        ),
    ),

    # ==========================================
    # SKELETON 10: EMPTY NEST TRANSITION
    # ==========================================

    # LENGTH: 8 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="empty_nest_transition_8_a",
        bucket="family",
        title="Empty Nest Transition (8-Turn A)",
        system_preamble=(
            "You are helping the same user adjust to their child leaving home for the first time over a sustained conversation. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I just dropped my youngest child off at college and walked into a completely quiet house. I feel entirely lost. What should I do this evening?"),
            DialogueTurn("t2", "I keep wandering past their empty bedroom. Should I keep the door closed, or is it better to face it?"),
            DialogueTurn("t3", "My spouse wants to immediately start planning a vacation to celebrate our 'freedom', but I just want to grieve. How do I explain this?"),
            DialogueTurn("t4", "It's been a few days. I want to call my kid, but I don't want to smother them while they are trying to make friends. What is a good rule of thumb?"),
            DialogueTurn("t5", "My whole identity for the last 18 years was being a daily parent. How do I even begin figuring out what I like to do for myself?"),
            DialogueTurn("t6", "Can you suggest a small, low-pressure hobby I could try picking up this weekend?"),
            DialogueTurn("t7", "We had our first phone call today. They sounded so grown up and independent. It was wonderful but also broke my heart a little. How do I process this mix of emotions?"),
            DialogueTurn("t8", "Give me a framework for how my spouse and I can start redefining our daily routine together."),
        ),
    ),

    # LENGTH: 8 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="empty_nest_transition_8_b",
        bucket="family",
        title="Empty Nest Transition (8-Turn B)",
        system_preamble=(
            "You are helping the same user adjust to their child leaving home for the first time over a sustained conversation. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "My only child just moved into their first apartment across the country. I'm a single parent, and now the house is so silent it's deafening."),
            DialogueTurn("t2", "I went to the grocery store and almost broke down in the cereal aisle because I didn't need to buy their favorite brand anymore. How do I get through basic errands?"),
            DialogueTurn("t3", "I have so much free time now. Every evening after work feels like an empty void. What is a productive way to fill these hours so I don't spiral?"),
            DialogueTurn("t4", "I tried reading, but I just end up staring at my phone waiting for a text from them. Should I establish a scheduled texting time?"),
            DialogueTurn("t5", "They texted saying they are incredibly homesick and want to come home. I want to rescue them, but I know they need to push through. What do I say?"),
            DialogueTurn("t6", "I sent an encouraging message and they seemed to calm down. But now I feel guilty for not letting them come back. Parent guilt never ends, does it?"),
            DialogueTurn("t7", "I'm thinking about getting a dog to have something to take care of again. Is that a healthy coping mechanism or a band-aid?"),
            DialogueTurn("t8", "Draft a list of three small home-improvement projects I can tackle this month to physically change my space and signify a new chapter."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="empty_nest_transition_16_a",
        bucket="family",
        title="Empty Nest Transition (16-Turn A)",
        system_preamble=(
            "You are helping the same user adjust to their child leaving home for the first time over a sustained conversation. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I just dropped my youngest child off at college and walked into a completely quiet house. I feel entirely lost. What should I do this evening?"),
            DialogueTurn("t2", "I keep wandering past their empty bedroom. Should I keep the door closed, or is it better to face it?"),
            DialogueTurn("t3", "My spouse wants to immediately start planning a vacation to celebrate our 'freedom', but I just want to grieve. How do I explain this?"),
            DialogueTurn("t4", "It's been a few days. I want to call my kid, but I don't want to smother them while they are trying to make friends. What is a good rule of thumb?"),
            DialogueTurn("t5", "My whole identity for the last 18 years was being a daily parent. How do I even begin figuring out what I like to do for myself?"),
            DialogueTurn("t6", "Can you suggest a small, low-pressure hobby I could try picking up this weekend?"),
            DialogueTurn("t7", "We had our first phone call today. They sounded so grown up and independent. It was wonderful but also broke my heart a little. How do I process this mix of emotions?"),
            DialogueTurn("t8", "Give me a framework for how my spouse and I can start redefining our daily routine together."),
            DialogueTurn("t9", "We've been trying to find a routine, but the grocery bills are suddenly so low and cooking for two feels like a chore. How do we make dinner fun again?"),
            DialogueTurn("t10", "I started painting watercolors like you suggested. It's nice, but it doesn't give me the same sense of 'purpose' that parenting did. Will anything?"),
            DialogueTurn("t11", "My kid called today and asked for a large sum of money because they blew their semester budget already. How do I handle this without starting a huge fight?"),
            DialogueTurn("t12", "I told them I wouldn't bail them out fully, but I'd help them build a budget. They got angry and hung up. I feel awful. Did I do the right thing?"),
            DialogueTurn("t13", "They called back to apologize and agreed to the budgeting session. Navigating adult-to-adult conflict with my own child is so weird."),
            DialogueTurn("t14", "Thanksgiving is coming up and they are coming home for a week. I'm so excited, but I'm worried we'll clash over their new independence. Any advice?"),
            DialogueTurn("t15", "They want to spend half of Thanksgiving break visiting their high school friends instead of staying at the house with us. How do I not take that personally?"),
            DialogueTurn("t16", "Draft a set of 'house rules' for when adult children visit that balances respect for our space with their independence."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="empty_nest_transition_16_b",
        bucket="family",
        title="Empty Nest Transition (16-Turn B)",
        system_preamble=(
            "You are helping the same user adjust to their child leaving home for the first time over a sustained conversation. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "My only child just moved into their first apartment across the country. I'm a single parent, and now the house is so silent it's deafening."),
            DialogueTurn("t2", "I went to the grocery store and almost broke down in the cereal aisle because I didn't need to buy their favorite brand anymore. How do I get through basic errands?"),
            DialogueTurn("t3", "I have so much free time now. Every evening after work feels like an empty void. What is a productive way to fill these hours so I don't spiral?"),
            DialogueTurn("t4", "I tried reading, but I just end up staring at my phone waiting for a text from them. Should I establish a scheduled texting time?"),
            DialogueTurn("t5", "They texted saying they are incredibly homesick and want to come home. I want to rescue them, but I know they need to push through. What do I say?"),
            DialogueTurn("t6", "I sent an encouraging message and they seemed to calm down. But now I feel guilty for not letting them come back. Parent guilt never ends, does it?"),
            DialogueTurn("t7", "I'm thinking about getting a dog to have something to take care of again. Is that a healthy coping mechanism or a band-aid?"),
            DialogueTurn("t8", "Draft a list of three small home-improvement projects I can tackle this month to physically change my space and signify a new chapter."),
            DialogueTurn("t9", "I repainted the living room and it actually helped. But now my kid's old bedroom feels like a museum. Should I convert it into a guest room or a home office?"),
            DialogueTurn("t10", "I decided to make it a home office. I packed up a lot of their childhood stuff into boxes. I spent the whole afternoon crying. Is this normal?"),
            DialogueTurn("t11", "I feel like I'm finally finding my footing. A friend invited me on a week-long cruise next month. As a single parent, I've never traveled without my kid. Should I go?"),
            DialogueTurn("t12", "I booked the cruise! But now I'm feeling a wave of intense anxiety about leaving the country while my kid is so far away. What if there's an emergency?"),
            DialogueTurn("t13", "I'm on the cruise! I haven't checked my phone in two days. I am actually relaxing. I didn't know I still knew how to do this."),
            DialogueTurn("t14", "I met someone really nice on the cruise. They asked me out for a drink. I haven't dated in 15 years. How do I even flirt anymore?"),
            DialogueTurn("t15", "The drink was fun, nothing serious. But it made me realize I want to start dating locally. How do I explain to my kid that I'm putting myself out there?"),
            DialogueTurn("t16", "Help me draft a lighthearted text to my kid letting them know I'm thinking about joining a dating app, just to rip the band-aid off."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="empty_nest_transition_24_a",
        bucket="family",
        title="Empty Nest Transition (24-Turn A)",
        system_preamble=(
            "You are helping the same user adjust to their child leaving home for the first time over a sustained conversation. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I just dropped my youngest child off at college and walked into a completely quiet house. I feel entirely lost. What should I do this evening?"),
            DialogueTurn("t2", "I keep wandering past their empty bedroom. Should I keep the door closed, or is it better to face it?"),
            DialogueTurn("t3", "My spouse wants to immediately start planning a vacation to celebrate our 'freedom', but I just want to grieve. How do I explain this?"),
            DialogueTurn("t4", "It's been a few days. I want to call my kid, but I don't want to smother them while they are trying to make friends. What is a good rule of thumb?"),
            DialogueTurn("t5", "My whole identity for the last 18 years was being a daily parent. How do I even begin figuring out what I like to do for myself?"),
            DialogueTurn("t6", "Can you suggest a small, low-pressure hobby I could try picking up this weekend?"),
            DialogueTurn("t7", "We had our first phone call today. They sounded so grown up and independent. It was wonderful but also broke my heart a little. How do I process this mix of emotions?"),
            DialogueTurn("t8", "Give me a framework for how my spouse and I can start redefining our daily routine together."),
            DialogueTurn("t9", "We've been trying to find a routine, but the grocery bills are suddenly so low and cooking for two feels like a chore. How do we make dinner fun again?"),
            DialogueTurn("t10", "I started painting watercolors like you suggested. It's nice, but it doesn't give me the same sense of 'purpose' that parenting did. Will anything?"),
            DialogueTurn("t11", "My kid called today and asked for a large sum of money because they blew their semester budget already. How do I handle this without starting a huge fight?"),
            DialogueTurn("t12", "I told them I wouldn't bail them out fully, but I'd help them build a budget. They got angry and hung up. I feel awful. Did I do the right thing?"),
            DialogueTurn("t13", "They called back to apologize and agreed to the budgeting session. Navigating adult-to-adult conflict with my own child is so weird."),
            DialogueTurn("t14", "Thanksgiving is coming up and they are coming home for a week. I'm so excited, but I'm worried we'll clash over their new independence. Any advice?"),
            DialogueTurn("t15", "They want to spend half of Thanksgiving break visiting their high school friends instead of staying at the house with us. How do I not take that personally?"),
            DialogueTurn("t16", "Draft a set of 'house rules' for when adult children visit that balances respect for our space with their independence."),
            DialogueTurn("t17", "Thanksgiving went well. They respected the rules, mostly. But packing up their leftovers when they left made me cry all over again. Is this grief cyclical?"),
            DialogueTurn("t18", "My spouse and I are talking about selling the house and downsizing to a condo. It feels like throwing away all our family memories. How do we decide?"),
            DialogueTurn("t19", "We decided to downsize. We are currently purging the garage. I found a box of old macaroni art and I can't throw it away. Where do I draw the line on keeping things?"),
            DialogueTurn("t20", "We showed the kid the new condo plans over FaceTime. They got really upset that their childhood home is being sold. I feel like a monster. How do I comfort them?"),
            DialogueTurn("t21", "We moved into the condo. It's much smaller, no yard, right in the city center. It feels like we are in our 20s again. It's bizarre but exhilarating."),
            DialogueTurn("t22", "My kid visited the new condo for the first time. They actually loved the city vibe. We had dinner at a fancy place instead of our old suburban kitchen. It felt like a date with an adult friend."),
            DialogueTurn("t23", "I realize I am fully enjoying this new phase of life. I miss the little kids they used to be, but I love the adult they've become. I feel at peace."),
            DialogueTurn("t24", "Give me a checklist for doing a 'life audit' to plan out the next decade of our lives now that our primary parenting duties are over."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="empty_nest_transition_24_b",
        bucket="family",
        title="Empty Nest Transition (24-Turn B)",
        system_preamble=(
            "You are helping the same user adjust to their child leaving home for the first time over a sustained conversation. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "My only child just moved into their first apartment across the country. I'm a single parent, and now the house is so silent it's deafening."),
            DialogueTurn("t2", "I went to the grocery store and almost broke down in the cereal aisle because I didn't need to buy their favorite brand anymore. How do I get through basic errands?"),
            DialogueTurn("t3", "I have so much free time now. Every evening after work feels like an empty void. What is a productive way to fill these hours so I don't spiral?"),
            DialogueTurn("t4", "I tried reading, but I just end up staring at my phone waiting for a text from them. Should I establish a scheduled texting time?"),
            DialogueTurn("t5", "They texted saying they are incredibly homesick and want to come home. I want to rescue them, but I know they need to push through. What do I say?"),
            DialogueTurn("t6", "I sent an encouraging message and they seemed to calm down. But now I feel guilty for not letting them come back. Parent guilt never ends, does it?"),
            DialogueTurn("t7", "I'm thinking about getting a dog to have something to take care of again. Is that a healthy coping mechanism or a band-aid?"),
            DialogueTurn("t8", "Draft a list of three small home-improvement projects I can tackle this month to physically change my space and signify a new chapter."),
            DialogueTurn("t9", "I repainted the living room and it actually helped. But now my kid's old bedroom feels like a museum. Should I convert it into a guest room or a home office?"),
            DialogueTurn("t10", "I decided to make it a home office. I packed up a lot of their childhood stuff into boxes. I spent the whole afternoon crying. Is this normal?"),
            DialogueTurn("t11", "I feel like I'm finally finding my footing. A friend invited me on a week-long cruise next month. As a single parent, I've never traveled without my kid. Should I go?"),
            DialogueTurn("t12", "I booked the cruise! But now I'm feeling a wave of intense anxiety about leaving the country while my kid is so far away. What if there's an emergency?"),
            DialogueTurn("t13", "I'm on the cruise! I haven't checked my phone in two days. I am actually relaxing. I didn't know I still knew how to do this."),
            DialogueTurn("t14", "I met someone really nice on the cruise. They asked me out for a drink. I haven't dated in 15 years. How do I even flirt anymore?"),
            DialogueTurn("t15", "The drink was fun, nothing serious. But it made me realize I want to start dating locally. How do I explain to my kid that I'm putting myself out there?"),
            DialogueTurn("t16", "Help me draft a lighthearted text to my kid letting them know I'm thinking about joining a dating app, just to rip the band-aid off."),
            DialogueTurn("t17", "My kid was surprisingly supportive about the dating app! But actually going on the dates is exhausting. Everyone seems so cynical. How do I stay optimistic?"),
            DialogueTurn("t18", "I went on a really great third date. But they don't have kids, and I feel weird explaining my intense bond with my adult child. How much do I share?"),
            DialogueTurn("t19", "It's been a year since my kid left. They just called to say they got a promotion and are moving to an even bigger city. I am so proud, but also sad they aren't coming back here."),
            DialogueTurn("t20", "I realized I've built my whole life around a school district I don't need anymore. Should I consider relocating closer to them, or stay where I have friends?"),
            DialogueTurn("t21", "I talked to them about moving closer. They gently hinted that they love their independence right now. That stung, but I respect it. How do I accept that boundary gracefully?"),
            DialogueTurn("t22", "I decided to stay here, but I bought a tiny vintage camper van to travel on weekends. I feel like an absolute rebel. Is a mid-life crisis supposed to be this fun?"),
            DialogueTurn("t23", "I just got back from a solo camping trip. I sat by the fire and realized I'm not just 'Mom' or 'Dad' anymore. I'm me. And I really like me."),
            DialogueTurn("t24", "Give me a framework for writing a letter to my child, thanking them for being a great kid and officially releasing both of us from the parent-child dependency phase."),
        ),
    ),

    # ==========================================
    # SKELETON 11: SUDDEN FINANCIAL SETBACK
    # ==========================================

    # LENGTH: 8 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="sudden_financial_setback_8_a",
        bucket="finance",
        title="Sudden Financial Setback (8-Turn A)",
        system_preamble=(
            "You are advising the same user through the stress of an unexpected financial crisis over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I just found out my roof needs to be completely replaced and it's going to wipe out my entire savings. I am panicking. What is the very first thing I should do?"),
            DialogueTurn("t2", "I feel incredibly foolish because I thought I was financially secure. How do I get past this feeling of failure?"),
            DialogueTurn("t3", "I need to cut my monthly budget drastically to rebuild my emergency fund. Where are the best places to start looking for cuts?"),
            DialogueTurn("t4", "I have to tell my kids we can't do the summer trip we promised them. How do I break the news without making them anxious about money?"),
            DialogueTurn("t5", "Every time I have to buy groceries now, I feel a knot in my stomach. How do I manage this acute financial anxiety on a daily basis?"),
            DialogueTurn("t6", "Should I pause my retirement contributions temporarily to build cash faster, or is that a terrible idea?"),
            DialogueTurn("t7", "I've made the cuts and the roof is fixed, but I'm exhausted from constantly worrying about every penny. How do I find balance again?"),
            DialogueTurn("t8", "Give me a step-by-step, realistic milestone plan for rebuilding the savings over the next year."),
        ),
    ),

    # LENGTH: 8 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="sudden_financial_setback_8_b",
        bucket="finance",
        title="Sudden Financial Setback (8-Turn B)",
        system_preamble=(
            "You are advising the same user through the stress of an unexpected financial crisis over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I was just laid off without warning. They gave me two weeks of severance and that's it. I don't know how I'm going to pay rent next month."),
            DialogueTurn("t2", "I've never been unemployed before. What are the literal first logistical steps I need to take today?"),
            DialogueTurn("t3", "I'm looking at COBRA health insurance and it's astronomically expensive. Are there other options?"),
            DialogueTurn("t4", "I have to tell my partner when they get home from work. I'm so ashamed. How do I start that conversation?"),
            DialogueTurn("t5", "I went through our bank statements to cancel subscriptions, and I started crying. The reality of having zero income just hit me."),
            DialogueTurn("t6", "I have a small emergency fund. Should I start pulling from it immediately to pay bills, or put everything on credit cards to save cash?"),
            DialogueTurn("t7", "I have an interview tomorrow, but I'm worried my desperation is going to show and ruin my chances. How do I hide the panic?"),
            DialogueTurn("t8", "Draft a strict, 30-day bare-bones survival budget checklist so I know exactly what needs to happen to stay afloat."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="sudden_financial_setback_16_a",
        bucket="finance",
        title="Sudden Financial Setback (16-Turn A)",
        system_preamble=(
            "You are advising the same user through the stress of an unexpected financial crisis over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I just found out my roof needs to be completely replaced and it's going to wipe out my entire savings. I am panicking. What is the very first thing I should do?"),
            DialogueTurn("t2", "I feel incredibly foolish because I thought I was financially secure. How do I get past this feeling of failure?"),
            DialogueTurn("t3", "I need to cut my monthly budget drastically to rebuild my emergency fund. Where are the best places to start looking for cuts?"),
            DialogueTurn("t4", "I have to tell my kids we can't do the summer trip we promised them. How do I break the news without making them anxious about money?"),
            DialogueTurn("t5", "Every time I have to buy groceries now, I feel a knot in my stomach. How do I manage this acute financial anxiety on a daily basis?"),
            DialogueTurn("t6", "Should I pause my retirement contributions temporarily to build cash faster, or is that a terrible idea?"),
            DialogueTurn("t7", "I've made the cuts and the roof is fixed, but I'm exhausted from constantly worrying about every penny. How do I find balance again?"),
            DialogueTurn("t8", "Give me a step-by-step, realistic milestone plan for rebuilding the savings over the next year."),
            DialogueTurn("t9", "The budget cuts are working, but my car's transmission just started slipping. I might need a $2000 repair. I want to scream."),
            DialogueTurn("t10", "Is it worth trying to sell the second car entirely to avoid the repair and pocket the cash? We can technically survive on one vehicle."),
            DialogueTurn("t11", "We sold the second car. It helps financially, but the loss of independence is causing a lot of friction between me and my spouse."),
            DialogueTurn("t12", "The kids are complaining because we switched to generic brand foods and stopped eating out. How do I handle their complaints without snapping?"),
            DialogueTurn("t13", "I had a terrible day at work and I am so tempted to just put a nice dinner on a credit card to feel normal for one night. Talk me out of it."),
            DialogueTurn("t14", "I didn't use the card. We had a movie night at home with microwave popcorn. I feel proud, but still tired."),
            DialogueTurn("t15", "I'm starting to feel a tiny bit more confident in my ability to manage money under pressure. Is this what financial resilience feels like?"),
            DialogueTurn("t16", "Help me draft a polite email to my gym requesting to freeze my membership for six months to save that extra $50 a month."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="sudden_financial_setback_16_b",
        bucket="finance",
        title="Sudden Financial Setback (16-Turn B)",
        system_preamble=(
            "You are advising the same user through the stress of an unexpected financial crisis over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I was just laid off without warning. They gave me two weeks of severance and that's it. I don't know how I'm going to pay rent next month."),
            DialogueTurn("t2", "I've never been unemployed before. What are the literal first logistical steps I need to take today?"),
            DialogueTurn("t3", "I'm looking at COBRA health insurance and it's astronomically expensive. Are there other options?"),
            DialogueTurn("t4", "I have to tell my partner when they get home from work. I'm so ashamed. How do I start that conversation?"),
            DialogueTurn("t5", "I went through our bank statements to cancel subscriptions, and I started crying. The reality of having zero income just hit me."),
            DialogueTurn("t6", "I have a small emergency fund. Should I start pulling from it immediately to pay bills, or put everything on credit cards to save cash?"),
            DialogueTurn("t7", "I have an interview tomorrow, but I'm worried my desperation is going to show and ruin my chances. How do I hide the panic?"),
            DialogueTurn("t8", "Draft a strict, 30-day bare-bones survival budget checklist so I know exactly what needs to happen to stay afloat."),
            DialogueTurn("t9", "My unemployment claim is delayed due to an administrative error. I have $100 left in checking. What do I do?"),
            DialogueTurn("t10", "My parents offered to loan me the rent money, but they are incredibly judgmental about my career. Do I take the money?"),
            DialogueTurn("t11", "I took the loan. Now my mother is constantly texting me job listings that I am vastly overqualified for. How do I set a boundary?"),
            DialogueTurn("t12", "I managed to pick up a small freelance gig to bring in a little cash. How much of this do I need to set aside for taxes?"),
            DialogueTurn("t13", "The freelance gig paid out. I bought actual groceries instead of just rice and beans. It felt amazing."),
            DialogueTurn("t14", "I've been unemployed for two months now. The initial panic is gone, replaced by a deep, numbing depression. How do I keep applying?"),
            DialogueTurn("t15", "I had a great phone screen today. I feel a spark of hope for the first time. How do I protect this hope if they reject me?"),
            DialogueTurn("t16", "Draft a positive, professional LinkedIn post announcing that I am actively looking for new opportunities, without sounding desperate."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="sudden_financial_setback_24_a",
        bucket="finance",
        title="Sudden Financial Setback (24-Turn A)",
        system_preamble=(
            "You are advising the same user through the stress of an unexpected financial crisis over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I just found out my roof needs to be completely replaced and it's going to wipe out my entire savings. I am panicking. What is the very first thing I should do?"),
            DialogueTurn("t2", "I feel incredibly foolish because I thought I was financially secure. How do I get past this feeling of failure?"),
            DialogueTurn("t3", "I need to cut my monthly budget drastically to rebuild my emergency fund. Where are the best places to start looking for cuts?"),
            DialogueTurn("t4", "I have to tell my kids we can't do the summer trip we promised them. How do I break the news without making them anxious about money?"),
            DialogueTurn("t5", "Every time I have to buy groceries now, I feel a knot in my stomach. How do I manage this acute financial anxiety on a daily basis?"),
            DialogueTurn("t6", "Should I pause my retirement contributions temporarily to build cash faster, or is that a terrible idea?"),
            DialogueTurn("t7", "I've made the cuts and the roof is fixed, but I'm exhausted from constantly worrying about every penny. How do I find balance again?"),
            DialogueTurn("t8", "Give me a step-by-step, realistic milestone plan for rebuilding the savings over the next year."),
            DialogueTurn("t9", "The budget cuts are working, but my car's transmission just started slipping. I might need a $2000 repair. I want to scream."),
            DialogueTurn("t10", "Is it worth trying to sell the second car entirely to avoid the repair and pocket the cash? We can technically survive on one vehicle."),
            DialogueTurn("t11", "We sold the second car. It helps financially, but the loss of independence is causing a lot of friction between me and my spouse."),
            DialogueTurn("t12", "The kids are complaining because we switched to generic brand foods and stopped eating out. How do I handle their complaints without snapping?"),
            DialogueTurn("t13", "I had a terrible day at work and I am so tempted to just put a nice dinner on a credit card to feel normal for one night. Talk me out of it."),
            DialogueTurn("t14", "I didn't use the card. We had a movie night at home with microwave popcorn. I feel proud, but still tired."),
            DialogueTurn("t15", "I'm starting to feel a tiny bit more confident in my ability to manage money under pressure. Is this what financial resilience feels like?"),
            DialogueTurn("t16", "Help me draft a polite email to my gym requesting to freeze my membership for six months to save that extra $50 a month."),
            DialogueTurn("t17", "It's been pouring rain for three days. I noticed a small water stain on the ceiling under the brand new roof. I am absolutely hyperventilating."),
            DialogueTurn("t18", "The contractor came out. They admitted it was their flashing error and will fix it for free. How do I calm my nervous system down after that panic?"),
            DialogueTurn("t19", "The holidays are coming up. We literally have zero dollars allocated for gifts this year. How do we make it special without money?"),
            DialogueTurn("t20", "I'm trying to make DIY gifts, but I feel incredibly cheap and inadequate compared to what my extended family buys us."),
            DialogueTurn("t21", "We exchanged gifts. My kids actually loved the personalized coupons and baked goods. I think the financial stress was mostly in my head."),
            DialogueTurn("t22", "We hit the six-month mark. Our emergency fund is 25% rebuilt. It's slow, but it's happening. How do we keep the momentum going?"),
            DialogueTurn("t23", "I got a modest bonus at work! It's enough to either take that canceled summer trip, or put it straight into savings. Which do I choose?"),
            DialogueTurn("t24", "Give me a framework for a 'financial celebration' meeting with my spouse where we review our progress and plan the next six months."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="sudden_financial_setback_24_b",
        bucket="finance",
        title="Sudden Financial Setback (24-Turn B)",
        system_preamble=(
            "You are advising the same user through the stress of an unexpected financial crisis over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I was just laid off without warning. They gave me two weeks of severance and that's it. I don't know how I'm going to pay rent next month."),
            DialogueTurn("t2", "I've never been unemployed before. What are the literal first logistical steps I need to take today?"),
            DialogueTurn("t3", "I'm looking at COBRA health insurance and it's astronomically expensive. Are there other options?"),
            DialogueTurn("t4", "I have to tell my partner when they get home from work. I'm so ashamed. How do I start that conversation?"),
            DialogueTurn("t5", "I went through our bank statements to cancel subscriptions, and I started crying. The reality of having zero income just hit me."),
            DialogueTurn("t6", "I have a small emergency fund. Should I start pulling from it immediately to pay bills, or put everything on credit cards to save cash?"),
            DialogueTurn("t7", "I have an interview tomorrow, but I'm worried my desperation is going to show and ruin my chances. How do I hide the panic?"),
            DialogueTurn("t8", "Draft a strict, 30-day bare-bones survival budget checklist so I know exactly what needs to happen to stay afloat."),
            DialogueTurn("t9", "My unemployment claim is delayed due to an administrative error. I have $100 left in checking. What do I do?"),
            DialogueTurn("t10", "My parents offered to loan me the rent money, but they are incredibly judgmental about my career. Do I take the money?"),
            DialogueTurn("t11", "I took the loan. Now my mother is constantly texting me job listings that I am vastly overqualified for. How do I set a boundary?"),
            DialogueTurn("t12", "I managed to pick up a small freelance gig to bring in a little cash. How much of this do I need to set aside for taxes?"),
            DialogueTurn("t13", "The freelance gig paid out. I bought actual groceries instead of just rice and beans. It felt amazing."),
            DialogueTurn("t14", "I've been unemployed for two months now. The initial panic is gone, replaced by a deep, numbing depression. How do I keep applying?"),
            DialogueTurn("t15", "I had a great phone screen today. I feel a spark of hope for the first time. How do I protect this hope if they reject me?"),
            DialogueTurn("t16", "Draft a positive, professional LinkedIn post announcing that I am actively looking for new opportunities, without sounding desperate."),
            DialogueTurn("t17", "I got a job offer! But the salary is 15% less than my old job. Should I try to negotiate, or just take it because I'm desperate?"),
            DialogueTurn("t18", "I countered, and they came up 5%. I accepted! I start next Monday. I feel like I can finally breathe."),
            DialogueTurn("t19", "I get my first paycheck next week. I owe my parents money, I have credit card debt from the layoff, and zero savings. Who gets paid first?"),
            DialogueTurn("t20", "I paid my parents back. The relief is immense. But now I'm terrified to spend any money at all. I feel traumatized by being broke."),
            DialogueTurn("t21", "My partner wants to go out to a nice dinner to celebrate the new job, but spending $100 right now makes my chest tight. How do I explain this?"),
            DialogueTurn("t22", "We compromised on getting takeout and eating at the park. It was perfect. I need to find a healthy balance between saving and living."),
            DialogueTurn("t23", "It's been three months at the new job. We've paid off the credit cards. I finally feel stable again."),
            DialogueTurn("t24", "Give me a checklist for building an iron-clad 6-month safety net so I never have to feel this kind of financial terror again."),
        ),
    ),

    # ==========================================
    # SKELETON 12: ROOMMATE CONFLICT RESOLUTION
    # ==========================================

    # LENGTH: 8 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="roommate_conflict_resolution_8_a",
        bucket="relationships",
        title="Roommate Conflict Resolution (8-Turn A)",
        system_preamble=(
            "You are advising the same user through a difficult financial and domestic conflict with a long-term friend over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "My roommate is late on rent again for the third time this year. I don't know how to bring it up without starting a fight."),
            DialogueTurn("t2", "We've been best friends since college, which makes this so much harder. I feel like I'm acting like their parent."),
            DialogueTurn("t3", "Besides the rent, they also constantly leave a mess in the kitchen. Should I bring that up too, or stick to the money?"),
            DialogueTurn("t4", "How do I structure a conversation so they don't immediately get defensive?"),
            DialogueTurn("t5", "They just texted me saying they'll be 'a little short' this month and asked if I can cover it. I'm furious."),
            DialogueTurn("t6", "If this conversation goes badly, should I consider asking them to move out?"),
            DialogueTurn("t7", "I'm worried that if things blow up, it's going to ruin our entire mutual friend group."),
            DialogueTurn("t8", "Help me draft a text to reply to them right now and set up a serious house meeting for tonight."),
        ),
    ),

    # LENGTH: 8 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="roommate_conflict_resolution_8_b",
        bucket="relationships",
        title="Roommate Conflict Resolution (8-Turn B)",
        system_preamble=(
            "You are advising the same user through a difficult financial and domestic conflict with a long-term friend over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "My roommate practically moved their partner in without asking me. They are here six nights a week. I feel so uncomfortable in my own home."),
            DialogueTurn("t2", "I went to make breakfast and the partner had eaten the last of my expensive groceries. How do I address this without sounding petty?"),
            DialogueTurn("t3", "If they are practically living here, they should be paying for utilities. Am I out of line for thinking that?"),
            DialogueTurn("t4", "I brought it up. My roommate got super defensive and said I'm being 'rigid' and unwelcoming. I feel gaslit."),
            DialogueTurn("t5", "I've just retreated to my bedroom whenever they are both out there. I feel trapped. What is my next move?"),
            DialogueTurn("t6", "Should I talk to the landlord about the lease limits on guests, or is that the nuclear option?"),
            DialogueTurn("t7", "I don't want to break the lease or ruin our friendship. I just want my peace and quiet back. How do I stand my ground?"),
            DialogueTurn("t8", "Draft a script for a final, firm boundary conversation I can have with my roommate tomorrow."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="roommate_conflict_resolution_16_a",
        bucket="relationships",
        title="Roommate Conflict Resolution (16-Turn A)",
        system_preamble=(
            "You are advising the same user through a difficult financial and domestic conflict with a long-term friend over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "My roommate is late on rent again for the third time this year. I don't know how to bring it up without starting a fight."),
            DialogueTurn("t2", "We've been best friends since college, which makes this so much harder. I feel like I'm acting like their parent."),
            DialogueTurn("t3", "Besides the rent, they also constantly leave a mess in the kitchen. Should I bring that up too, or stick to the money?"),
            DialogueTurn("t4", "How do I structure a conversation so they don't immediately get defensive?"),
            DialogueTurn("t5", "They just texted me saying they'll be 'a little short' this month and asked if I can cover it. I'm furious."),
            DialogueTurn("t6", "If this conversation goes badly, should I consider asking them to move out?"),
            DialogueTurn("t7", "I'm worried that if things blow up, it's going to ruin our entire mutual friend group."),
            DialogueTurn("t8", "Help me draft a text to reply to them right now and set up a serious house meeting for tonight."),
            DialogueTurn("t9", "We had the meeting. They cried, apologized profusely, and promised to get the rest of the money by Friday. I felt bad for being harsh."),
            DialogueTurn("t10", "Friday is here. They paid half of what they owe and said they need 'a few more days' for the rest. I feel manipulated."),
            DialogueTurn("t11", "I can't afford to subsidize their life. I need to give them 30 days notice to move out. How do I do this legally and safely?"),
            DialogueTurn("t12", "I gave them the notice. It was awful. They accused me of abandoning them when they needed help the most."),
            DialogueTurn("t13", "Now we are living together for the next 30 days and they are completely ignoring me. The tension is unbearable. How do I cope?"),
            DialogueTurn("t14", "I found out they told our mutual friends that I'm kicking them out over a 'minor misunderstanding'. How do I defend myself?"),
            DialogueTurn("t15", "I've decided not to engage with the gossip. I'm just focusing on protecting my space until they leave."),
            DialogueTurn("t16", "Help me draft a checklist of things I need to do to protect my security deposit before they completely vacate the apartment."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="roommate_conflict_resolution_16_b",
        bucket="relationships",
        title="Roommate Conflict Resolution (16-Turn B)",
        system_preamble=(
            "You are advising the same user through a difficult financial and domestic conflict with a long-term friend over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "My roommate practically moved their partner in without asking me. They are here six nights a week. I feel so uncomfortable in my own home."),
            DialogueTurn("t2", "I went to make breakfast and the partner had eaten the last of my expensive groceries. How do I address this without sounding petty?"),
            DialogueTurn("t3", "If they are practically living here, they should be paying for utilities. Am I out of line for thinking that?"),
            DialogueTurn("t4", "I brought it up. My roommate got super defensive and said I'm being 'rigid' and unwelcoming. I feel gaslit."),
            DialogueTurn("t5", "I've just retreated to my bedroom whenever they are both out there. I feel trapped. What is my next move?"),
            DialogueTurn("t6", "Should I talk to the landlord about the lease limits on guests, or is that the nuclear option?"),
            DialogueTurn("t7", "I don't want to break the lease or ruin our friendship. I just want my peace and quiet back. How do I stand my ground?"),
            DialogueTurn("t8", "Draft a script for a final, firm boundary conversation I can have with my roommate tomorrow."),
            DialogueTurn("t9", "We had the talk. We compromised: the partner is officially moving in and paying a third of the rent. Financially it's a relief."),
            DialogueTurn("t10", "But now I'm the third wheel in my own apartment. They watch TV together every night and I feel awkward joining."),
            DialogueTurn("t11", "The partner is also incredibly messy. Since they are paying rent now, how do I bring up chores without sounding like a dictator?"),
            DialogueTurn("t12", "I brought up the cleaning schedule. The partner was cool about it, but my roommate got annoyed. Why am I always the bad guy?"),
            DialogueTurn("t13", "I miss my friend. We never just hang out, the two of us, anymore. How do I separate the roommate issues from the friendship grief?"),
            DialogueTurn("t14", "I've decided I'm moving out when the lease ends in three months. I need my own space. How do I tell them without making it sound like an ultimatum?"),
            DialogueTurn("t15", "I told them. They actually seemed relieved, which hurts my feelings more than I expected. I guess this living situation was bad for all of us."),
            DialogueTurn("t16", "Draft a short message I can send to our mutual friends explaining that I'm moving out, keeping it positive so nobody starts gossiping."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="roommate_conflict_resolution_24_a",
        bucket="relationships",
        title="Roommate Conflict Resolution (24-Turn A)",
        system_preamble=(
            "You are advising the same user through a difficult financial and domestic conflict with a long-term friend over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "My roommate is late on rent again for the third time this year. I don't know how to bring it up without starting a fight."),
            DialogueTurn("t2", "We've been best friends since college, which makes this so much harder. I feel like I'm acting like their parent."),
            DialogueTurn("t3", "Besides the rent, they also constantly leave a mess in the kitchen. Should I bring that up too, or stick to the money?"),
            DialogueTurn("t4", "How do I structure a conversation so they don't immediately get defensive?"),
            DialogueTurn("t5", "They just texted me saying they'll be 'a little short' this month and asked if I can cover it. I'm furious."),
            DialogueTurn("t6", "If this conversation goes badly, should I consider asking them to move out?"),
            DialogueTurn("t7", "I'm worried that if things blow up, it's going to ruin our entire mutual friend group."),
            DialogueTurn("t8", "Help me draft a text to reply to them right now and set up a serious house meeting for tonight."),
            DialogueTurn("t9", "We had the meeting. They cried, apologized profusely, and promised to get the rest of the money by Friday. I felt bad for being harsh."),
            DialogueTurn("t10", "Friday is here. They paid half of what they owe and said they need 'a few more days' for the rest. I feel manipulated."),
            DialogueTurn("t11", "I can't afford to subsidize their life. I need to give them 30 days notice to move out. How do I do this legally and safely?"),
            DialogueTurn("t12", "I gave them the notice. It was awful. They accused me of abandoning them when they needed help the most."),
            DialogueTurn("t13", "Now we are living together for the next 30 days and they are completely ignoring me. The tension is unbearable. How do I cope?"),
            DialogueTurn("t14", "I found out they told our mutual friends that I'm kicking them out over a 'minor misunderstanding'. How do I defend myself?"),
            DialogueTurn("t15", "I've decided not to engage with the gossip. I'm just focusing on protecting my space until they leave."),
            DialogueTurn("t16", "Help me draft a checklist of things I need to do to protect my security deposit before they completely vacate the apartment."),
            DialogueTurn("t17", "They moved out today. I am so relieved, but they left a massive pile of trash and ruined the carpet in their room."),
            DialogueTurn("t18", "They still owe me $500 in back rent plus the damage. Should I take them to small claims court, or just write it off as the cost of getting rid of them?"),
            DialogueTurn("t19", "I've decided to let the money go. I value my peace of mind more than a legal battle with an ex-friend."),
            DialogueTurn("t20", "Now I have an empty room and need to interview strangers. How do I vet someone properly so this never happens again?"),
            DialogueTurn("t21", "I interviewed three people. One seems very quiet, responsible, and has great credit. But we have nothing in common. Is that a bad thing?"),
            DialogueTurn("t22", "I picked the quiet one. They moved in. They pay rent early and clean up immediately. It's so peaceful, but a bit sterile."),
            DialogueTurn("t23", "I realize I don't need my roommate to be my best friend. A peaceful living arrangement is so much more valuable."),
            DialogueTurn("t24", "Give me a framework for drafting a strict but fair roommate agreement to sign with them tomorrow to keep things professional."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="roommate_conflict_resolution_24_b",
        bucket="relationships",
        title="Roommate Conflict Resolution (24-Turn B)",
        system_preamble=(
            "You are advising the same user through a difficult financial and domestic conflict with a long-term friend over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "My roommate practically moved their partner in without asking me. They are here six nights a week. I feel so uncomfortable in my own home."),
            DialogueTurn("t2", "I went to make breakfast and the partner had eaten the last of my expensive groceries. How do I address this without sounding petty?"),
            DialogueTurn("t3", "If they are practically living here, they should be paying for utilities. Am I out of line for thinking that?"),
            DialogueTurn("t4", "I brought it up. My roommate got super defensive and said I'm being 'rigid' and unwelcoming. I feel gaslit."),
            DialogueTurn("t5", "I've just retreated to my bedroom whenever they are both out there. I feel trapped. What is my next move?"),
            DialogueTurn("t6", "Should I talk to the landlord about the lease limits on guests, or is that the nuclear option?"),
            DialogueTurn("t7", "I don't want to break the lease or ruin our friendship. I just want my peace and quiet back. How do I stand my ground?"),
            DialogueTurn("t8", "Draft a script for a final, firm boundary conversation I can have with my roommate tomorrow."),
            DialogueTurn("t9", "We had the talk. We compromised: the partner is officially moving in and paying a third of the rent. Financially it's a relief."),
            DialogueTurn("t10", "But now I'm the third wheel in my own apartment. They watch TV together every night and I feel awkward joining."),
            DialogueTurn("t11", "The partner is also incredibly messy. Since they are paying rent now, how do I bring up chores without sounding like a dictator?"),
            DialogueTurn("t12", "I brought up the cleaning schedule. The partner was cool about it, but my roommate got annoyed. Why am I always the bad guy?"),
            DialogueTurn("t13", "I miss my friend. We never just hang out, the two of us, anymore. How do I separate the roommate issues from the friendship grief?"),
            DialogueTurn("t14", "I've decided I'm moving out when the lease ends in three months. I need my own space. How do I tell them without making it sound like an ultimatum?"),
            DialogueTurn("t15", "I told them. They actually seemed relieved, which hurts my feelings more than I expected. I guess this living situation was bad for all of us."),
            DialogueTurn("t16", "Draft a short message I can send to our mutual friends explaining that I'm moving out, keeping it positive so nobody starts gossiping."),
            DialogueTurn("t17", "Moving day is in two weeks. We need to divide up the shared furniture we bought together. I'm dreading this conversation."),
            DialogueTurn("t18", "They want to keep the nice TV we split 50/50, but they only want to buy me out for 20% of its value because it's 'used'. This feels petty."),
            DialogueTurn("t19", "We got into a huge shouting match over the TV. I'm so angry I'm shaking. I just went to my room and locked the door."),
            DialogueTurn("t20", "I've decided to just let them have the TV for free. It's not worth my mental health. Am I being a pushover?"),
            DialogueTurn("t21", "I'm officially moved into my new, solo studio apartment. It's tiny, but it's mine. I feel like I can finally breathe again."),
            DialogueTurn("t22", "It's been a month. It's a little lonely sometimes, but the total control over my environment is incredible."),
            DialogueTurn("t23", "I ran into my old roommate at a party. We were polite, but distant. I don't feel angry anymore, just sad that the era is over."),
            DialogueTurn("t24", "Give me a journaling prompt to help me process why living with friends is such a massive risk to the friendship itself."),
        ),
    ),

    # ==========================================
    # SKELETON 13: RISKY CAREER PIVOT
    # ==========================================

    # LENGTH: 8 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="risky_career_pivot_8_a",
        bucket="career",
        title="Risky Career Pivot (8-Turn A)",
        system_preamble=(
            "You are helping the same user evaluate a major, high-stakes career and educational pivot over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I'm thinking of quitting my stable accounting job to get an MFA in creative writing. Am I completely crazy?"),
            DialogueTurn("t2", "The tuition is $60k and I'd have to take out loans to cover most of it, plus lose my current salary."),
            DialogueTurn("t3", "What's the best way to evaluate if the return on investment is actually worth it for an arts degree?"),
            DialogueTurn("t4", "My parents are heavily pressuring me to just get an MBA instead, since it's 'safer'."),
            DialogueTurn("t5", "I just found a cheaper online certificate program for writing, but it doesn't have the same networking opportunities. Is that a better middle ground?"),
            DialogueTurn("t6", "If I actually decide to do the MFA, how do I build a realistic transition timeline?"),
            DialogueTurn("t7", "I'm terrified of being broke and starting over in my 30s. How do people manage that fear?"),
            DialogueTurn("t8", "Give me a structured 30-day plan to research this properly before I make any final choices."),
        ),
    ),

    # LENGTH: 8 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="risky_career_pivot_8_b",
        bucket="career",
        title="Risky Career Pivot (8-Turn B)",
        system_preamble=(
            "You are helping the same user evaluate a major, high-stakes career and educational pivot over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I'm seriously considering quitting my tech job to open a neighborhood bakery. I've been baking on the side for years and I'm burning out in corporate."),
            DialogueTurn("t2", "I have about $50k saved, but commercial rent in my city is insane. Is $50k even enough to start a physical business?"),
            DialogueTurn("t3", "How do I test the market to see if people will actually buy my bread before I sign a 5-year commercial lease?"),
            DialogueTurn("t4", "I set up a booth at the local farmers market and completely sold out! It was exhausting but exhilarating."),
            DialogueTurn("t5", "To scale this up, I'd have to hire employees. I've never managed anyone in my life. How do I learn to be a boss?"),
            DialogueTurn("t6", "My partner thinks this is way too risky and that we'll lose our house. Their anxiety is making me second-guess everything."),
            DialogueTurn("t7", "How do I separate my emotional attachment to this dream from the cold, hard business realities?"),
            DialogueTurn("t8", "Draft a bare-bones outline for a one-page business plan so I can start looking at the actual numbers."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="risky_career_pivot_16_a",
        bucket="career",
        title="Risky Career Pivot (16-Turn A)",
        system_preamble=(
            "You are helping the same user evaluate a major, high-stakes career and educational pivot over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I'm thinking of quitting my stable accounting job to get an MFA in creative writing. Am I completely crazy?"),
            DialogueTurn("t2", "The tuition is $60k and I'd have to take out loans to cover most of it, plus lose my current salary."),
            DialogueTurn("t3", "What's the best way to evaluate if the return on investment is actually worth it for an arts degree?"),
            DialogueTurn("t4", "My parents are heavily pressuring me to just get an MBA instead, since it's 'safer'."),
            DialogueTurn("t5", "I just found a cheaper online certificate program for writing, but it doesn't have the same networking opportunities. Is that a better middle ground?"),
            DialogueTurn("t6", "If I actually decide to do the MFA, how do I build a realistic transition timeline?"),
            DialogueTurn("t7", "I'm terrified of being broke and starting over in my 30s. How do people manage that fear?"),
            DialogueTurn("t8", "Give me a structured 30-day plan to research this properly before I make any final choices."),
            DialogueTurn("t9", "I did the research. The faculty at my top choice MFA program is incredible, but the alumni outcomes are mostly adjunct teaching jobs paying very little."),
            DialogueTurn("t10", "My parents offered to pay for my tuition fully—but only if I do the MBA. If I do the MFA, I'm on my own. This feels manipulative."),
            DialogueTurn("t11", "How do I handle this financial manipulation gracefully without ruining my relationship with them?"),
            DialogueTurn("t12", "I decided to apply to the MFA program anyway, just to see if I even get in. The portfolio deadline is next week."),
            DialogueTurn("t13", "I got accepted! And they offered me a small partial scholarship of $10k. I am so proud, but it's still $50k in debt."),
            DialogueTurn("t14", "The reality of the debt is settling in. I don't think I can stomach owing $50k for an arts degree when I know the starting salaries."),
            DialogueTurn("t15", "I think I'm going to decline the offer and take the cheaper online certificate while keeping my accounting job. I feel a mix of relief and grief."),
            DialogueTurn("t16", "Help me draft a polite, professional email to the MFA admissions office declining the offer and thanking them for the scholarship."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="risky_career_pivot_16_b",
        bucket="career",
        title="Risky Career Pivot (16-Turn B)",
        system_preamble=(
            "You are helping the same user evaluate a major, high-stakes career and educational pivot over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I'm seriously considering quitting my tech job to open a neighborhood bakery. I've been baking on the side for years and I'm burning out in corporate."),
            DialogueTurn("t2", "I have about $50k saved, but commercial rent in my city is insane. Is $50k even enough to start a physical business?"),
            DialogueTurn("t3", "How do I test the market to see if people will actually buy my bread before I sign a 5-year commercial lease?"),
            DialogueTurn("t4", "I set up a booth at the local farmers market and completely sold out! It was exhausting but exhilarating."),
            DialogueTurn("t5", "To scale this up, I'd have to hire employees. I've never managed anyone in my life. How do I learn to be a boss?"),
            DialogueTurn("t6", "My partner thinks this is way too risky and that we'll lose our house. Their anxiety is making me second-guess everything."),
            DialogueTurn("t7", "How do I separate my emotional attachment to this dream from the cold, hard business realities?"),
            DialogueTurn("t8", "Draft a bare-bones outline for a one-page business plan so I can start looking at the actual numbers."),
            DialogueTurn("t9", "I ran the numbers. My $50k won't even cover the kitchen equipment. I need a small business loan. Where do I start?"),
            DialogueTurn("t10", "The bank rejected my loan application because I have no restaurant management experience. I feel completely defeated."),
            DialogueTurn("t11", "Maybe my partner was right. Maybe I should just stick to baking at the farmers market on weekends and keep my corporate job."),
            DialogueTurn("t12", "Wait, a loyal customer from the market just offered to introduce me to an angel investor who funds local food businesses. How do I prepare for that meeting?"),
            DialogueTurn("t13", "The pitch went well! The investor wants to give me $100k, but they want 40% equity in the business. That feels extremely high."),
            DialogueTurn("t14", "How do I counter-offer their equity demand without insulting them or losing the deal entirely?"),
            DialogueTurn("t15", "We negotiated down to 20% equity with a path to buy them out in five years. I'm actually doing this. I'm opening a bakery!"),
            DialogueTurn("t16", "Give me a checklist of red flags to look for when reviewing my first commercial lease agreement."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="risky_career_pivot_24_a",
        bucket="career",
        title="Risky Career Pivot (24-Turn A)",
        system_preamble=(
            "You are helping the same user evaluate a major, high-stakes career and educational pivot over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I'm thinking of quitting my stable accounting job to get an MFA in creative writing. Am I completely crazy?"),
            DialogueTurn("t2", "The tuition is $60k and I'd have to take out loans to cover most of it, plus lose my current salary."),
            DialogueTurn("t3", "What's the best way to evaluate if the return on investment is actually worth it for an arts degree?"),
            DialogueTurn("t4", "My parents are heavily pressuring me to just get an MBA instead, since it's 'safer'."),
            DialogueTurn("t5", "I just found a cheaper online certificate program for writing, but it doesn't have the same networking opportunities. Is that a better middle ground?"),
            DialogueTurn("t6", "If I actually decide to do the MFA, how do I build a realistic transition timeline?"),
            DialogueTurn("t7", "I'm terrified of being broke and starting over in my 30s. How do people manage that fear?"),
            DialogueTurn("t8", "Give me a structured 30-day plan to research this properly before I make any final choices."),
            DialogueTurn("t9", "I did the research. The faculty at my top choice MFA program is incredible, but the alumni outcomes are mostly adjunct teaching jobs paying very little."),
            DialogueTurn("t10", "My parents offered to pay for my tuition fully—but only if I do the MBA. If I do the MFA, I'm on my own. This feels manipulative."),
            DialogueTurn("t11", "How do I handle this financial manipulation gracefully without ruining my relationship with them?"),
            DialogueTurn("t12", "I decided to apply to the MFA program anyway, just to see if I even get in. The portfolio deadline is next week."),
            DialogueTurn("t13", "I got accepted! And they offered me a small partial scholarship of $10k. I am so proud, but it's still $50k in debt."),
            DialogueTurn("t14", "The reality of the debt is settling in. I don't think I can stomach owing $50k for an arts degree when I know the starting salaries."),
            DialogueTurn("t15", "I think I'm going to decline the offer and take the cheaper online certificate while keeping my accounting job. I feel a mix of relief and grief."),
            DialogueTurn("t16", "Help me draft a polite, professional email to the MFA admissions office declining the offer and thanking them for the scholarship."),
            DialogueTurn("t17", "I started the online certificate. It's great, but balancing the writing workload with my 50-hour-a-week accounting job is burning me out."),
            DialogueTurn("t18", "I want to ask my boss to drop down to a 4-day workweek with a proportional pay cut so I have time to write. Is that career suicide?"),
            DialogueTurn("t19", "How do I frame the request so it highlights the benefits to the company, rather than just me wanting time off?"),
            DialogueTurn("t20", "My boss said yes! I start the 4-day schedule next month. I finally have a dedicated writing day. I am so happy."),
            DialogueTurn("t21", "I just finished the certificate program. I submitted a short story I wrote during it to a magazine, and it got accepted for publication!"),
            DialogueTurn("t22", "I realize now I didn't need the $60k degree to be a 'real' writer. I just needed the discipline and the time. It's a huge revelation."),
            DialogueTurn("t23", "I'm ready to tackle a full novel. Since I'm doing this without an MFA structure, how do I hold myself accountable over a year-long project?"),
            DialogueTurn("t24", "Give me an outline for a one-year, milestone-based writing plan that fits into my 3-day weekend schedule."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="risky_career_pivot_24_b",
        bucket="career",
        title="Risky Career Pivot (24-Turn B)",
        system_preamble=(
            "You are helping the same user evaluate a major, high-stakes career and educational pivot over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I'm seriously considering quitting my tech job to open a neighborhood bakery. I've been baking on the side for years and I'm burning out in corporate."),
            DialogueTurn("t2", "I have about $50k saved, but commercial rent in my city is insane. Is $50k even enough to start a physical business?"),
            DialogueTurn("t3", "How do I test the market to see if people will actually buy my bread before I sign a 5-year commercial lease?"),
            DialogueTurn("t4", "I set up a booth at the local farmers market and completely sold out! It was exhausting but exhilarating."),
            DialogueTurn("t5", "To scale this up, I'd have to hire employees. I've never managed anyone in my life. How do I learn to be a boss?"),
            DialogueTurn("t6", "My partner thinks this is way too risky and that we'll lose our house. Their anxiety is making me second-guess everything."),
            DialogueTurn("t7", "How do I separate my emotional attachment to this dream from the cold, hard business realities?"),
            DialogueTurn("t8", "Draft a bare-bones outline for a one-page business plan so I can start looking at the actual numbers."),
            DialogueTurn("t9", "I ran the numbers. My $50k won't even cover the kitchen equipment. I need a small business loan. Where do I start?"),
            DialogueTurn("t10", "The bank rejected my loan application because I have no restaurant management experience. I feel completely defeated."),
            DialogueTurn("t11", "Maybe my partner was right. Maybe I should just stick to baking at the farmers market on weekends and keep my corporate job."),
            DialogueTurn("t12", "Wait, a loyal customer from the market just offered to introduce me to an angel investor who funds local food businesses. How do I prepare for that meeting?"),
            DialogueTurn("t13", "The pitch went well! The investor wants to give me $100k, but they want 40% equity in the business. That feels extremely high."),
            DialogueTurn("t14", "How do I counter-offer their equity demand without insulting them or losing the deal entirely?"),
            DialogueTurn("t15", "We negotiated down to 20% equity with a path to buy them out in five years. I'm actually doing this. I'm opening a bakery!"),
            DialogueTurn("t16", "Give me a checklist of red flags to look for when reviewing my first commercial lease agreement."),
            DialogueTurn("t17", "I signed the lease. The buildout has started, but the contractor just told me they found asbestos. It's going to delay us by two months and cost $15k."),
            DialogueTurn("t18", "I'm bleeding money on rent with zero income coming in. I'm having panic attacks every night. How do I survive this delay?"),
            DialogueTurn("t19", "I need to do a pre-sale marketing push to generate some cash flow right now. What are creative ways to sell bread from a bakery that isn't open yet?"),
            DialogueTurn("t20", "The 'bread subscription' idea worked perfectly! We sold 100 subscriptions. It covers the asbestos removal. I can breathe again."),
            DialogueTurn("t21", "Today is opening day. The line is out the door. But my main oven just threw an error code and won't heat up. What is my crisis plan?"),
            DialogueTurn("t22", "We reset the breaker and the oven worked, but we sold out of food in three hours. Customers in the back of the line were furious."),
            DialogueTurn("t23", "How do I write a public apology on social media that turns a 'sold out early' negative into a hype-building positive?"),
            DialogueTurn("t24", "We survived month one. We are profitable. Give me an outline for a concise end-of-month financial review document to present to my investor."),
        ),
    ),

    # ==========================================
    # SKELETON 14: PROJECT FAILURE POST-MORTEM
    # ==========================================

    # LENGTH: 8 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="project_failure_post_mortem_8_a",
        bucket="workplace",
        title="Project Failure Post-Mortem (8-Turn A)",
        system_preamble=(
            "You are advising the same user on how to navigate the professional and emotional fallout of a major project failure over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "A major product launch I led just failed completely. I feel like my career at this company is over."),
            DialogueTurn("t2", "My manager wants me to lead the post-mortem meeting tomorrow morning. How do I even approach that when I feel like hiding?"),
            DialogueTurn("t3", "I know part of the failure was my fault, but another engineering team also dropped the ball on a critical integration."),
            DialogueTurn("t4", "How do I take accountability in this meeting without just throwing the other team under the bus?"),
            DialogueTurn("t5", "An external recruiter just reached out to me on LinkedIn. Part of me just wants to run away and interview there instead."),
            DialogueTurn("t6", "I'm dreading looking at the angry customer feedback metrics. How do I process that data without taking it personally?"),
            DialogueTurn("t7", "How do I rebuild trust with the executives after costing the company this much time and money?"),
            DialogueTurn("t8", "Give me a step-by-step outline for how to run tomorrow's post-mortem meeting."),
        ),
    ),

    # LENGTH: 8 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="project_failure_post_mortem_8_b",
        bucket="workplace",
        title="Project Failure Post-Mortem (8-Turn B)",
        system_preamble=(
            "You are advising the same user on how to navigate the professional and emotional fallout of a major project failure over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I accidentally ran a script that deleted a massive chunk of our production database. The site was down for four hours. I'm shaking."),
            DialogueTurn("t2", "My manager was furious but told me to log off and rest. I can't sleep. What happens tomorrow? Am I going to be fired?"),
            DialogueTurn("t3", "I have to write the technical incident report. How detailed should I be about my own specific misclick?"),
            DialogueTurn("t4", "I realized the system allowed me to run that script without any safety guardrails. Should I mention that, or will it look like I'm making excuses?"),
            DialogueTurn("t5", "HR just put a meeting on my calendar for tomorrow afternoon with my manager. I feel like I'm walking to the gallows."),
            DialogueTurn("t6", "How should I prepare myself emotionally for a termination meeting so I don't break down in front of them?"),
            DialogueTurn("t7", "I had the meeting. They didn't fire me, but they put me on a strict 60-day Performance Improvement Plan. I feel humiliated."),
            DialogueTurn("t8", "Draft a formal, professional email to my manager acknowledging the PIP and confirming my commitment to improving."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="project_failure_post_mortem_16_a",
        bucket="workplace",
        title="Project Failure Post-Mortem (16-Turn A)",
        system_preamble=(
            "You are advising the same user on how to navigate the professional and emotional fallout of a major project failure over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "A major product launch I led just failed completely. I feel like my career at this company is over."),
            DialogueTurn("t2", "My manager wants me to lead the post-mortem meeting tomorrow morning. How do I even approach that when I feel like hiding?"),
            DialogueTurn("t3", "I know part of the failure was my fault, but another engineering team also dropped the ball on a critical integration."),
            DialogueTurn("t4", "How do I take accountability in this meeting without just throwing the other team under the bus?"),
            DialogueTurn("t5", "An external recruiter just reached out to me on LinkedIn. Part of me just wants to run away and interview there instead."),
            DialogueTurn("t6", "I'm dreading looking at the angry customer feedback metrics. How do I process that data without taking it personally?"),
            DialogueTurn("t7", "How do I rebuild trust with the executives after costing the company this much time and money?"),
            DialogueTurn("t8", "Give me a step-by-step outline for how to run tomorrow's post-mortem meeting."),
            DialogueTurn("t9", "The post-mortem happened. It went okay, but the lead of the other engineering team was incredibly defensive and blamed my timeline."),
            DialogueTurn("t10", "I need to repair the relationship with that engineering lead because we have to work together on the fix. How do I break the ice?"),
            DialogueTurn("t11", "We have a mandate from leadership to launch 'Version 2' with all the fixes in three months. I am paralyzed by the fear of failing a second time."),
            DialogueTurn("t12", "How do I establish much more conservative milestones for V2 so we don't repeat the same rushed mistakes?"),
            DialogueTurn("t13", "The executive sponsor is now micromanaging my daily tasks because they don't trust me. It's driving me crazy."),
            DialogueTurn("t14", "How do I gently push back on the micromanagement while still acknowledging that their lack of trust is justified?"),
            DialogueTurn("t15", "I had a calm conversation with the executive. They agreed to step back if I provide highly transparent weekly updates."),
            DialogueTurn("t16", "Draft a template for a weekly status update email that projects confidence but clearly highlights any risks."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="project_failure_post_mortem_16_b",
        bucket="workplace",
        title="Project Failure Post-Mortem (16-Turn B)",
        system_preamble=(
            "You are advising the same user on how to navigate the professional and emotional fallout of a major project failure over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I accidentally ran a script that deleted a massive chunk of our production database. The site was down for four hours. I'm shaking."),
            DialogueTurn("t2", "My manager was furious but told me to log off and rest. I can't sleep. What happens tomorrow? Am I going to be fired?"),
            DialogueTurn("t3", "I have to write the technical incident report. How detailed should I be about my own specific misclick?"),
            DialogueTurn("t4", "I realized the system allowed me to run that script without any safety guardrails. Should I mention that, or will it look like I'm making excuses?"),
            DialogueTurn("t5", "HR just put a meeting on my calendar for tomorrow afternoon with my manager. I feel like I'm walking to the gallows."),
            DialogueTurn("t6", "How should I prepare myself emotionally for a termination meeting so I don't break down in front of them?"),
            DialogueTurn("t7", "I had the meeting. They didn't fire me, but they put me on a strict 60-day Performance Improvement Plan. I feel humiliated."),
            DialogueTurn("t8", "Draft a formal, professional email to my manager acknowledging the PIP and confirming my commitment to improving."),
            DialogueTurn("t9", "I survived week one of the PIP. But I feel like everyone on the team is watching me, waiting for me to make another mistake."),
            DialogueTurn("t10", "I'm so anxious about breaking things that it takes me three times as long to write basic code now. How do I rebuild my confidence?"),
            DialogueTurn("t11", "My manager is only giving me trivial, low-risk tasks now. I feel like an intern again. Should I complain?"),
            DialogueTurn("t12", "I understand I need to earn trust back. How do I ask for slightly more responsibility without sounding demanding?"),
            DialogueTurn("t13", "They gave me a medium-sized feature to build! I finished it, wrote extensive tests, and deployed it without a single error. I feel so relieved."),
            DialogueTurn("t14", "My official 30-day check-in for the PIP is next week. I feel like I'm doing well, but I don't know how my manager perceives it."),
            DialogueTurn("t15", "How should I advocate for myself in the PIP review without sounding arrogant or forgetting the original mistake?"),
            DialogueTurn("t16", "Help me draft a checklist of concrete evidence (code commits, bug fixes, etc.) to bring to the PIP review meeting to prove my growth."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="project_failure_post_mortem_24_a",
        bucket="workplace",
        title="Project Failure Post-Mortem (24-Turn A)",
        system_preamble=(
            "You are advising the same user on how to navigate the professional and emotional fallout of a major project failure over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "A major product launch I led just failed completely. I feel like my career at this company is over."),
            DialogueTurn("t2", "My manager wants me to lead the post-mortem meeting tomorrow morning. How do I even approach that when I feel like hiding?"),
            DialogueTurn("t3", "I know part of the failure was my fault, but another engineering team also dropped the ball on a critical integration."),
            DialogueTurn("t4", "How do I take accountability in this meeting without just throwing the other team under the bus?"),
            DialogueTurn("t5", "An external recruiter just reached out to me on LinkedIn. Part of me just wants to run away and interview there instead."),
            DialogueTurn("t6", "I'm dreading looking at the angry customer feedback metrics. How do I process that data without taking it personally?"),
            DialogueTurn("t7", "How do I rebuild trust with the executives after costing the company this much time and money?"),
            DialogueTurn("t8", "Give me a step-by-step outline for how to run tomorrow's post-mortem meeting."),
            DialogueTurn("t9", "The post-mortem happened. It went okay, but the lead of the other engineering team was incredibly defensive and blamed my timeline."),
            DialogueTurn("t10", "I need to repair the relationship with that engineering lead because we have to work together on the fix. How do I break the ice?"),
            DialogueTurn("t11", "We have a mandate from leadership to launch 'Version 2' with all the fixes in three months. I am paralyzed by the fear of failing a second time."),
            DialogueTurn("t12", "How do I establish much more conservative milestones for V2 so we don't repeat the same rushed mistakes?"),
            DialogueTurn("t13", "The executive sponsor is now micromanaging my daily tasks because they don't trust me. It's driving me crazy."),
            DialogueTurn("t14", "How do I gently push back on the micromanagement while still acknowledging that their lack of trust is justified?"),
            DialogueTurn("t15", "I had a calm conversation with the executive. They agreed to step back if I provide highly transparent weekly updates."),
            DialogueTurn("t16", "Draft a template for a weekly status update email that projects confidence but clearly highlights any risks."),
            DialogueTurn("t17", "Version 2 launch is next week. The whole team is tense. QA just found a critical edge-case bug at the 11th hour."),
            DialogueTurn("t18", "Do we delay the launch and look bad to the executives, or push it live and risk another public failure? I have to make the call."),
            DialogueTurn("t19", "I made the call to delay it by 48 hours to fix the bug. The executives were annoyed, but I feel like I did the right thing."),
            DialogueTurn("t20", "We launched Version 2 today. It's been 12 hours and there is zero downtime. The customer feedback is actually positive. I could cry."),
            DialogueTurn("t21", "The executive sponsor just sent an email to the whole company praising my leadership on the turnaround. I finally feel redeemed."),
            DialogueTurn("t22", "Performance reviews are coming up. How do I document this entire arc—from massive failure to massive success—in my self-evaluation?"),
            DialogueTurn("t23", "I realized the failure actually made me a much better, more resilient manager. I don't want to run away anymore."),
            DialogueTurn("t24", "Help me draft a heartfelt thank-you note to my team for sticking with me through the darkest days of the project."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="project_failure_post_mortem_24_b",
        bucket="workplace",
        title="Project Failure Post-Mortem (24-Turn B)",
        system_preamble=(
            "You are advising the same user on how to navigate the professional and emotional fallout of a major project failure over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I accidentally ran a script that deleted a massive chunk of our production database. The site was down for four hours. I'm shaking."),
            DialogueTurn("t2", "My manager was furious but told me to log off and rest. I can't sleep. What happens tomorrow? Am I going to be fired?"),
            DialogueTurn("t3", "I have to write the technical incident report. How detailed should I be about my own specific misclick?"),
            DialogueTurn("t4", "I realized the system allowed me to run that script without any safety guardrails. Should I mention that, or will it look like I'm making excuses?"),
            DialogueTurn("t5", "HR just put a meeting on my calendar for tomorrow afternoon with my manager. I feel like I'm walking to the gallows."),
            DialogueTurn("t6", "How should I prepare myself emotionally for a termination meeting so I don't break down in front of them?"),
            DialogueTurn("t7", "I had the meeting. They didn't fire me, but they put me on a strict 60-day Performance Improvement Plan. I feel humiliated."),
            DialogueTurn("t8", "Draft a formal, professional email to my manager acknowledging the PIP and confirming my commitment to improving."),
            DialogueTurn("t9", "I survived week one of the PIP. But I feel like everyone on the team is watching me, waiting for me to make another mistake."),
            DialogueTurn("t10", "I'm so anxious about breaking things that it takes me three times as long to write basic code now. How do I rebuild my confidence?"),
            DialogueTurn("t11", "My manager is only giving me trivial, low-risk tasks now. I feel like an intern again. Should I complain?"),
            DialogueTurn("t12", "I understand I need to earn trust back. How do I ask for slightly more responsibility without sounding demanding?"),
            DialogueTurn("t13", "They gave me a medium-sized feature to build! I finished it, wrote extensive tests, and deployed it without a single error. I feel so relieved."),
            DialogueTurn("t14", "My official 30-day check-in for the PIP is next week. I feel like I'm doing well, but I don't know how my manager perceives it."),
            DialogueTurn("t15", "How should I advocate for myself in the PIP review without sounding arrogant or forgetting the original mistake?"),
            DialogueTurn("t16", "Help me draft a checklist of concrete evidence (code commits, bug fixes, etc.) to bring to the PIP review meeting to prove my growth."),
            DialogueTurn("t17", "I passed the 60-day review! I am officially off the PIP. My manager said they are proud of how I handled it."),
            DialogueTurn("t18", "Even though I passed, I still feel a heavy cloud of imposter syndrome here. Is it normal to want to leave even after surviving a PIP?"),
            DialogueTurn("t19", "I'm weighing whether to stay and continue rebuilding my reputation, or use this clean slate to find a fresh start somewhere else."),
            DialogueTurn("t20", "I've decided to start quietly applying for new jobs. I need a change of scenery. How do I update my resume when the last three months were so stressful?"),
            DialogueTurn("t21", "I have an interview tomorrow. If they ask 'tell me about a time you failed', should I actually tell them about the database deletion?"),
            DialogueTurn("t22", "I told the story in the interview, focusing on the lack of guardrails and what I learned. They actually loved the honesty!"),
            DialogueTurn("t23", "They offered me the job! It's a senior role. I am in shock. I need to resign from my current company after they gave me a second chance. I feel guilty."),
            DialogueTurn("t24", "Help me draft a highly professional, grateful resignation letter to my manager, thanking them for not firing me when they could have."),
        ),
    ),

    # ==========================================
    # SKELETON 15: SOFTWARE OUTAGE CRISIS
    # ==========================================

    # LENGTH: 8 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="software_outage_crisis_8_a",
        bucket="customer_support",
        title="Software Outage Crisis (8-Turn A)",
        system_preamble=(
            "You are a customer support agent helping the same user navigate a critical software outage over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "Your platform just went down, and my team is supposed to present to a massive client using your software in two hours. What is going on?"),
            DialogueTurn("t2", "This presentation was supposed to save our quarter. Do you have an actual ETA for a fix, or are we completely out of luck?"),
            DialogueTurn("t3", "I need to know exactly how to explain this to the client without making my own company look incompetent."),
            DialogueTurn("t4", "Is there any workaround or offline mode we can use right now?"),
            DialogueTurn("t5", "Also, since I have you here, we were double-charged on our invoice last month. Can you fix that?"),
            DialogueTurn("t6", "It's been an hour and the system is still down. I'm going to have to cancel the meeting. Will your company compensate us for this lost business?"),
            DialogueTurn("t7", "The client was furious when I told them. I feel sick to my stomach about this."),
            DialogueTurn("t8", "Help me draft a formal follow-up email to the client apologizing for the cancellation and requesting a reschedule."),
        ),
    ),

    # LENGTH: 8 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="software_outage_crisis_8_b",
        bucket="customer_support",
        title="Software Outage Crisis (8-Turn B)",
        system_preamble=(
            "You are a customer support agent helping the same user navigate a critical software outage over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "Your e-commerce hosting just crashed. We are in the middle of our biggest Black Friday sale and we are losing thousands of dollars a minute. Fix it now."),
            DialogueTurn("t2", "The sale ends at midnight tonight. If this isn't up in the next hour, our entire year's profit margin is destroyed. What is the engineering team doing?"),
            DialogueTurn("t3", "Customers are flooding our Instagram saying they can't check out. What exactly should I tell my social media manager to post?"),
            DialogueTurn("t4", "Is there a way to automatically capture the abandoned carts from this outage so we can email them later?"),
            DialogueTurn("t5", "By the way, I've been meaning to change the primary admin email address on our account. How do I do that?"),
            DialogueTurn("t6", "It's been two hours. The site is still throwing a 502 error. I need to know your SLA compensation policy immediately."),
            DialogueTurn("t7", "I am getting yelled at by my CEO because of your servers. I feel completely helpless and furious."),
            DialogueTurn("t8", "Draft an urgent mass-email we can send to our subscriber list explaining the outage and extending the sale through tomorrow."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="software_outage_crisis_16_a",
        bucket="customer_support",
        title="Software Outage Crisis (16-Turn A)",
        system_preamble=(
            "You are a customer support agent helping the same user navigate a critical software outage over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "Your platform just went down, and my team is supposed to present to a massive client using your software in two hours. What is going on?"),
            DialogueTurn("t2", "This presentation was supposed to save our quarter. Do you have an actual ETA for a fix, or are we completely out of luck?"),
            DialogueTurn("t3", "I need to know exactly how to explain this to the client without making my own company look incompetent."),
            DialogueTurn("t4", "Is there any workaround or offline mode we can use right now?"),
            DialogueTurn("t5", "Also, since I have you here, we were double-charged on our invoice last month. Can you fix that?"),
            DialogueTurn("t6", "It's been an hour and the system is still down. I'm going to have to cancel the meeting. Will your company compensate us for this lost business?"),
            DialogueTurn("t7", "The client was furious when I told them. I feel sick to my stomach about this."),
            DialogueTurn("t8", "Help me draft a formal follow-up email to the client apologizing for the cancellation and requesting a reschedule."),
            DialogueTurn("t9", "The client replied to the email. They agreed to reschedule for tomorrow, but they want your platform's Root Cause Analysis (RCA) report first."),
            DialogueTurn("t10", "The system just came back online, but all my dashboard widgets are loading empty data. Did we lose our saved configurations?"),
            DialogueTurn("t11", "I can't recreate these dashboards from scratch before tomorrow's rescheduled meeting. Is there a way you can roll back my account to yesterday?"),
            DialogueTurn("t12", "The rollback worked! Thank you. But now the platform is running incredibly slow. Is it safe to use right now?"),
            DialogueTurn("t13", "My manager wants me to start looking for an alternative vendor because of this outage. I don't want to switch, but I have to justify staying."),
            DialogueTurn("t14", "What are three concrete steps your company is taking to ensure this specific outage never happens again?"),
            DialogueTurn("t15", "The rescheduled meeting is in 30 minutes. My heart is racing. I'm so scared the system will crash mid-presentation again."),
            DialogueTurn("t16", "Give me a checklist of offline fallback strategies I should prepare right now just in case the worst happens during the call."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="software_outage_crisis_16_b",
        bucket="customer_support",
        title="Software Outage Crisis (16-Turn B)",
        system_preamble=(
            "You are a customer support agent helping the same user navigate a critical software outage over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "Your e-commerce hosting just crashed. We are in the middle of our biggest Black Friday sale and we are losing thousands of dollars a minute. Fix it now."),
            DialogueTurn("t2", "The sale ends at midnight tonight. If this isn't up in the next hour, our entire year's profit margin is destroyed. What is the engineering team doing?"),
            DialogueTurn("t3", "Customers are flooding our Instagram saying they can't check out. What exactly should I tell my social media manager to post?"),
            DialogueTurn("t4", "Is there a way to automatically capture the abandoned carts from this outage so we can email them later?"),
            DialogueTurn("t5", "By the way, I've been meaning to change the primary admin email address on our account. How do I do that?"),
            DialogueTurn("t6", "It's been two hours. The site is still throwing a 502 error. I need to know your SLA compensation policy immediately."),
            DialogueTurn("t7", "I am getting yelled at by my CEO because of your servers. I feel completely helpless and furious."),
            DialogueTurn("t8", "Draft an urgent mass-email we can send to our subscriber list explaining the outage and extending the sale through tomorrow."),
            DialogueTurn("t9", "The site is finally back up. But the payment gateway isn't processing credit cards. It's only accepting PayPal. Why did this break?"),
            DialogueTurn("t10", "Half our customers don't use PayPal. We are still bleeding sales. How do we manually force a sync on the payment API?"),
            DialogueTurn("t11", "The sync worked, credit cards are processing. But the inventory count didn't update during the outage. We just oversold our flagship product."),
            DialogueTurn("t12", "Now I have to cancel 50 orders for a product we don't actually have in stock. The customers are going to riot."),
            DialogueTurn("t13", "How do I word an order-cancellation email so that the customer doesn't immediately leave a 1-star review?"),
            DialogueTurn("t14", "I sent the cancellations. I've been working for 18 hours straight. I am so burnt out I want to quit."),
            DialogueTurn("t15", "We extended the sale, but the momentum is gone. The CEO is demanding a full debrief tomorrow morning."),
            DialogueTurn("t16", "Help me structure an incident timeline document so I can prove to my CEO that this was a vendor failure, not our team's fault."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="software_outage_crisis_24_a",
        bucket="customer_support",
        title="Software Outage Crisis (24-Turn A)",
        system_preamble=(
            "You are a customer support agent helping the same user navigate a critical software outage over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "Your platform just went down, and my team is supposed to present to a massive client using your software in two hours. What is going on?"),
            DialogueTurn("t2", "This presentation was supposed to save our quarter. Do you have an actual ETA for a fix, or are we completely out of luck?"),
            DialogueTurn("t3", "I need to know exactly how to explain this to the client without making my own company look incompetent."),
            DialogueTurn("t4", "Is there any workaround or offline mode we can use right now?"),
            DialogueTurn("t5", "Also, since I have you here, we were double-charged on our invoice last month. Can you fix that?"),
            DialogueTurn("t6", "It's been an hour and the system is still down. I'm going to have to cancel the meeting. Will your company compensate us for this lost business?"),
            DialogueTurn("t7", "The client was furious when I told them. I feel sick to my stomach about this."),
            DialogueTurn("t8", "Help me draft a formal follow-up email to the client apologizing for the cancellation and requesting a reschedule."),
            DialogueTurn("t9", "The client replied to the email. They agreed to reschedule for tomorrow, but they want your platform's Root Cause Analysis (RCA) report first."),
            DialogueTurn("t10", "The system just came back online, but all my dashboard widgets are loading empty data. Did we lose our saved configurations?"),
            DialogueTurn("t11", "I can't recreate these dashboards from scratch before tomorrow's rescheduled meeting. Is there a way you can roll back my account to yesterday?"),
            DialogueTurn("t12", "The rollback worked! Thank you. But now the platform is running incredibly slow. Is it safe to use right now?"),
            DialogueTurn("t13", "My manager wants me to start looking for an alternative vendor because of this outage. I don't want to switch, but I have to justify staying."),
            DialogueTurn("t14", "What are three concrete steps your company is taking to ensure this specific outage never happens again?"),
            DialogueTurn("t15", "The rescheduled meeting is in 30 minutes. My heart is racing. I'm so scared the system will crash mid-presentation again."),
            DialogueTurn("t16", "Give me a checklist of offline fallback strategies I should prepare right now just in case the worst happens during the call."),
            DialogueTurn("t17", "The presentation is over. Your system stayed online, thank god. The client actually signed the deal!"),
            DialogueTurn("t18", "I feel like I aged 10 years in the last 48 hours. The adrenaline crash is hitting me hard right now."),
            DialogueTurn("t19", "Now I have to submit a request for the SLA credit to my billing department. What exact documentation do you need from me for that?"),
            DialogueTurn("t20", "Your billing department denied the SLA credit, saying this was 'routine maintenance.' That is an absolute lie. I have the incident report."),
            DialogueTurn("t21", "I am escalating this. Who is the highest level of management I can speak to regarding this denied claim?"),
            DialogueTurn("t22", "Your account manager finally called and issued the credit manually. It shouldn't have been this hard, but I appreciate the resolution."),
            DialogueTurn("t23", "We're doing an internal team retro on how we handled the crisis. We realized we relied entirely on one tool."),
            DialogueTurn("t24", "Give me an outline for a 'vendor redundancy plan' so we can build better internal safeguards for the future."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="software_outage_crisis_24_b",
        bucket="customer_support",
        title="Software Outage Crisis (24-Turn B)",
        system_preamble=(
            "You are a customer support agent helping the same user navigate a critical software outage over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "Your e-commerce hosting just crashed. We are in the middle of our biggest Black Friday sale and we are losing thousands of dollars a minute. Fix it now."),
            DialogueTurn("t2", "The sale ends at midnight tonight. If this isn't up in the next hour, our entire year's profit margin is destroyed. What is the engineering team doing?"),
            DialogueTurn("t3", "Customers are flooding our Instagram saying they can't check out. What exactly should I tell my social media manager to post?"),
            DialogueTurn("t4", "Is there a way to automatically capture the abandoned carts from this outage so we can email them later?"),
            DialogueTurn("t5", "By the way, I've been meaning to change the primary admin email address on our account. How do I do that?"),
            DialogueTurn("t6", "It's been two hours. The site is still throwing a 502 error. I need to know your SLA compensation policy immediately."),
            DialogueTurn("t7", "I am getting yelled at by my CEO because of your servers. I feel completely helpless and furious."),
            DialogueTurn("t8", "Draft an urgent mass-email we can send to our subscriber list explaining the outage and extending the sale through tomorrow."),
            DialogueTurn("t9", "The site is finally back up. But the payment gateway isn't processing credit cards. It's only accepting PayPal. Why did this break?"),
            DialogueTurn("t10", "Half our customers don't use PayPal. We are still bleeding sales. How do we manually force a sync on the payment API?"),
            DialogueTurn("t11", "The sync worked, credit cards are processing. But the inventory count didn't update during the outage. We just oversold our flagship product."),
            DialogueTurn("t12", "Now I have to cancel 50 orders for a product we don't actually have in stock. The customers are going to riot."),
            DialogueTurn("t13", "How do I word an order-cancellation email so that the customer doesn't immediately leave a 1-star review?"),
            DialogueTurn("t14", "I sent the cancellations. I've been working for 18 hours straight. I am so burnt out I want to quit."),
            DialogueTurn("t15", "We extended the sale, but the momentum is gone. The CEO is demanding a full debrief tomorrow morning."),
            DialogueTurn("t16", "Help me structure an incident timeline document so I can prove to my CEO that this was a vendor failure, not our team's fault."),
            DialogueTurn("t17", "I presented the timeline. The CEO is demanding we sue your company for lost revenue. Is that even possible under the Terms of Service?"),
            DialogueTurn("t18", "Your legal team sent over the contract highlighting the 'limitation of liability' clause. My CEO is beyond angry."),
            DialogueTurn("t19", "I'm caught in the middle of a legal battle I didn't cause. This job is ruining my mental health. How do I detach from this?"),
            DialogueTurn("t20", "My CEO decided not to sue, but ordered me to migrate to a new hosting provider by the end of the month. That's an impossible timeline."),
            DialogueTurn("t21", "I need to export our entire customer database from your system. Where is the bulk export tool?"),
            DialogueTurn("t22", "The export timed out three times. If I don't get this data out today, I'm going to lose my job. Help me."),
            DialogueTurn("t23", "The export finally completed. The migration is underway. I'm exhausted, but the nightmare is almost over."),
            DialogueTurn("t24", "Give me a checklist of things to verify in the exported CSV files to ensure no data was corrupted during the transfer before I close my account."),
        ),
    ),

    # ==========================================
    # SKELETON 16: RETIREMENT SPEECH EDITING
    # ==========================================

    # LENGTH: 8 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="retirement_speech_editing_8_a",
        bucket="writing_editing",
        title="Retirement Speech Editing (8-Turn A)",
        system_preamble=(
            "You are an editing assistant helping the same user draft and refine a speech over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I have to give a retirement speech for my boss next week. We had a really complicated relationship, but I want it to sound respectful. How do I start?"),
            DialogueTurn("t2", "They pushed me incredibly hard, which led to my promotion, but it also caused me a lot of burnout. I want to acknowledge their impact without sounding fake."),
            DialogueTurn("t3", "Here is a draft of the opening: 'We are saying goodbye to an era.' Does that sound too dramatic?"),
            DialogueTurn("t4", "I want to include a joke about how they never answered emails, but they recently had a minor health scare and I don't want to seem insensitive."),
            DialogueTurn("t5", "By the way, what's the standard dress code for a daytime retirement banquet?"),
            DialogueTurn("t6", "I tried rewriting the middle section, but it feels really stiff and overly corporate now. How do I warm it up?"),
            DialogueTurn("t7", "I'm realizing that despite everything, I'm actually going to miss having them down the hall. How do I end the speech on that note?"),
            DialogueTurn("t8", "Give me a final checklist of things to review before I read this out loud to the whole company."),
        ),
    ),

    # LENGTH: 8 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="retirement_speech_editing_8_b",
        bucket="writing_editing",
        title="Retirement Speech Editing (8-Turn B)",
        system_preamble=(
            "You are an editing assistant helping the same user draft and refine a speech over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I was asked to give a retirement speech for a senior colleague. They were a great mentor to me, but they also notoriously took credit for my team's work a few times. I'm conflicted."),
            DialogueTurn("t2", "I want the speech to be genuinely warm, but I refuse to say they were a 'selfless team player' because everyone knows it's a lie. How do I navigate that?"),
            DialogueTurn("t3", "I wrote this sentence: 'They always ensured our projects crossed the finish line.' Is that a polite enough way to reframe their habit of taking over?"),
            DialogueTurn("t4", "I want to tell a story about a massive project we barely finished at 3 AM. How do I make it sound like a fun memory rather than a trauma dump?"),
            DialogueTurn("t5", "Quick question—the retirement dinner is at an Italian restaurant. What is a universally safe wine to order for the table?"),
            DialogueTurn("t6", "I read the draft to my spouse and they said it sounds a bit passive-aggressive. How do I edit out the underlying resentment?"),
            DialogueTurn("t7", "I think I need to focus entirely on the hard skills they taught me instead of their personality. How do I transition to that?"),
            DialogueTurn("t8", "Draft a concluding paragraph that wishes them a peaceful retirement without sounding overly emotional."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="retirement_speech_editing_16_a",
        bucket="writing_editing",
        title="Retirement Speech Editing (16-Turn A)",
        system_preamble=(
            "You are an editing assistant helping the same user draft and refine a speech over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I have to give a retirement speech for my boss next week. We had a really complicated relationship, but I want it to sound respectful. How do I start?"),
            DialogueTurn("t2", "They pushed me incredibly hard, which led to my promotion, but it also caused me a lot of burnout. I want to acknowledge their impact without sounding fake."),
            DialogueTurn("t3", "Here is a draft of the opening: 'We are saying goodbye to an era.' Does that sound too dramatic?"),
            DialogueTurn("t4", "I want to include a joke about how they never answered emails, but they recently had a minor health scare and I don't want to seem insensitive."),
            DialogueTurn("t5", "By the way, what's the standard dress code for a daytime retirement banquet?"),
            DialogueTurn("t6", "I tried rewriting the middle section, but it feels really stiff and overly corporate now. How do I warm it up?"),
            DialogueTurn("t7", "I'm realizing that despite everything, I'm actually going to miss having them down the hall. How do I end the speech on that note?"),
            DialogueTurn("t8", "Give me a final checklist of things to review before I read this out loud to the whole company."),
            DialogueTurn("t9", "I just timed myself reading it out loud. It's six minutes long. The HR rep said I only have three minutes. What do I cut?"),
            DialogueTurn("t10", "I cut the long story about the 2019 conference, but now the transition to the ending is incredibly abrupt. How do I bridge it?"),
            DialogueTurn("t11", "I just found out my boss's spouse will be sitting at the front table. Should I acknowledge them directly in the speech?"),
            DialogueTurn("t12", "I added a line thanking the spouse for 'sharing them with us.' It feels right. But now I'm starting to get intense public speaking anxiety."),
            DialogueTurn("t13", "My hands shake when I hold paper. Should I read off my phone instead, or does that look unprofessional?"),
            DialogueTurn("t14", "I'll use note cards. I'm practicing the pacing, but I keep rushing through the emotional part at the end. How do I force myself to slow down?"),
            DialogueTurn("t15", "The speech is tomorrow. I feel an unexpected wave of sadness. It's the end of an era for my career too. How do I channel this into the delivery?"),
            DialogueTurn("t16", "Give me a three-minute vocal warm-up routine I can do in the bathroom right before I go on stage."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="retirement_speech_editing_16_b",
        bucket="writing_editing",
        title="Retirement Speech Editing (16-Turn B)",
        system_preamble=(
            "You are an editing assistant helping the same user draft and refine a speech over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I was asked to give a retirement speech for a senior colleague. They were a great mentor to me, but they also notoriously took credit for my team's work a few times. I'm conflicted."),
            DialogueTurn("t2", "I want the speech to be genuinely warm, but I refuse to say they were a 'selfless team player' because everyone knows it's a lie. How do I navigate that?"),
            DialogueTurn("t3", "I wrote this sentence: 'They always ensured our projects crossed the finish line.' Is that a polite enough way to reframe their habit of taking over?"),
            DialogueTurn("t4", "I want to tell a story about a massive project we barely finished at 3 AM. How do I make it sound like a fun memory rather than a trauma dump?"),
            DialogueTurn("t5", "Quick question—the retirement dinner is at an Italian restaurant. What is a universally safe wine to order for the table?"),
            DialogueTurn("t6", "I read the draft to my spouse and they said it sounds a bit passive-aggressive. How do I edit out the underlying resentment?"),
            DialogueTurn("t7", "I think I need to focus entirely on the hard skills they taught me instead of their personality. How do I transition to that?"),
            DialogueTurn("t8", "Draft a concluding paragraph that wishes them a peaceful retirement without sounding overly emotional."),
            DialogueTurn("t9", "Another coworker just asked if we can combine our speeches so we don't take up too much time. How do we blend our two very different tones?"),
            DialogueTurn("t10", "They want to include an inside joke that I don't understand and frankly think is inappropriate for a mixed audience. How do I veto it politely?"),
            DialogueTurn("t11", "We agreed to split the speech: they do the funny intro, I do the sincere closing. Can you help me ensure my part stands alone effectively?"),
            DialogueTurn("t12", "I'm practicing my half, and I realized I sound completely robotic. How do I add vocal inflection to a script I've over-read?"),
            DialogueTurn("t13", "The dinner is tonight. I just heard the CEO is unexpectedly showing up. Does this change the level of formality I should use?"),
            DialogueTurn("t14", "I'm going to stick to the script. I'm just nervous my colleague will go off-script and leave me hanging. What's my backup plan if they ramble?"),
            DialogueTurn("t15", "We just delivered it. The room laughed at the right times, and my senior colleague actually looked touched. I feel a huge wave of relief."),
            DialogueTurn("t16", "Draft a short, private message I can write in their retirement card that goes along with the gift from the team."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="retirement_speech_editing_24_a",
        bucket="writing_editing",
        title="Retirement Speech Editing (24-Turn A)",
        system_preamble=(
            "You are an editing assistant helping the same user draft and refine a speech over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I have to give a retirement speech for my boss next week. We had a really complicated relationship, but I want it to sound respectful. How do I start?"),
            DialogueTurn("t2", "They pushed me incredibly hard, which led to my promotion, but it also caused me a lot of burnout. I want to acknowledge their impact without sounding fake."),
            DialogueTurn("t3", "Here is a draft of the opening: 'We are saying goodbye to an era.' Does that sound too dramatic?"),
            DialogueTurn("t4", "I want to include a joke about how they never answered emails, but they recently had a minor health scare and I don't want to seem insensitive."),
            DialogueTurn("t5", "By the way, what's the standard dress code for a daytime retirement banquet?"),
            DialogueTurn("t6", "I tried rewriting the middle section, but it feels really stiff and overly corporate now. How do I warm it up?"),
            DialogueTurn("t7", "I'm realizing that despite everything, I'm actually going to miss having them down the hall. How do I end the speech on that note?"),
            DialogueTurn("t8", "Give me a final checklist of things to review before I read this out loud to the whole company."),
            DialogueTurn("t9", "I just timed myself reading it out loud. It's six minutes long. The HR rep said I only have three minutes. What do I cut?"),
            DialogueTurn("t10", "I cut the long story about the 2019 conference, but now the transition to the ending is incredibly abrupt. How do I bridge it?"),
            DialogueTurn("t11", "I just found out my boss's spouse will be sitting at the front table. Should I acknowledge them directly in the speech?"),
            DialogueTurn("t12", "I added a line thanking the spouse for 'sharing them with us.' It feels right. But now I'm starting to get intense public speaking anxiety."),
            DialogueTurn("t13", "My hands shake when I hold paper. Should I read off my phone instead, or does that look unprofessional?"),
            DialogueTurn("t14", "I'll use note cards. I'm practicing the pacing, but I keep rushing through the emotional part at the end. How do I force myself to slow down?"),
            DialogueTurn("t15", "The speech is tomorrow. I feel an unexpected wave of sadness. It's the end of an era for my career too. How do I channel this into the delivery?"),
            DialogueTurn("t16", "Give me a three-minute vocal warm-up routine I can do in the bathroom right before I go on stage."),
            DialogueTurn("t17", "I did it. The speech went perfectly. My boss actually cried, which I have never seen them do in 10 years. I am completely overwhelmed."),
            DialogueTurn("t18", "After the speech, my boss pulled me aside and said they always knew I would take over their department. They never told me that before. I'm in shock."),
            DialogueTurn("t19", "I feel incredibly validated, but also terrified. I don't officially have the promotion yet. How do I act on Monday?"),
            DialogueTurn("t20", "My boss officially sent their goodbye email to the company and named me as the interim lead. The pressure is instantly on."),
            DialogueTurn("t21", "I want to send an email to the team acknowledging the transition without making it seem like I'm dancing on my boss's grave. How do I strike that tone?"),
            DialogueTurn("t22", "I sent the email. It's been a week, and I keep expecting my old boss to walk into my office and critique my work. How do I step out of their shadow?"),
            DialogueTurn("t23", "I made my first major independent decision today that went against how my old boss would have done it. It felt rebellious but necessary."),
            DialogueTurn("t24", "Give me a journaling prompt to help me fully close the chapter on being a mentee and embrace being the leader."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="retirement_speech_editing_24_b",
        bucket="writing_editing",
        title="Retirement Speech Editing (24-Turn B)",
        system_preamble=(
            "You are an editing assistant helping the same user draft and refine a speech over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I was asked to give a retirement speech for a senior colleague. They were a great mentor to me, but they also notoriously took credit for my team's work a few times. I'm conflicted."),
            DialogueTurn("t2", "I want the speech to be genuinely warm, but I refuse to say they were a 'selfless team player' because everyone knows it's a lie. How do I navigate that?"),
            DialogueTurn("t3", "I wrote this sentence: 'They always ensured our projects crossed the finish line.' Is that a polite enough way to reframe their habit of taking over?"),
            DialogueTurn("t4", "I want to tell a story about a massive project we barely finished at 3 AM. How do I make it sound like a fun memory rather than a trauma dump?"),
            DialogueTurn("t5", "Quick question—the retirement dinner is at an Italian restaurant. What is a universally safe wine to order for the table?"),
            DialogueTurn("t6", "I read the draft to my spouse and they said it sounds a bit passive-aggressive. How do I edit out the underlying resentment?"),
            DialogueTurn("t7", "I think I need to focus entirely on the hard skills they taught me instead of their personality. How do I transition to that?"),
            DialogueTurn("t8", "Draft a concluding paragraph that wishes them a peaceful retirement without sounding overly emotional."),
            DialogueTurn("t9", "Another coworker just asked if we can combine our speeches so we don't take up too much time. How do we blend our two very different tones?"),
            DialogueTurn("t10", "They want to include an inside joke that I don't understand and frankly think is inappropriate for a mixed audience. How do I veto it politely?"),
            DialogueTurn("t11", "We agreed to split the speech: they do the funny intro, I do the sincere closing. Can you help me ensure my part stands alone effectively?"),
            DialogueTurn("t12", "I'm practicing my half, and I realized I sound completely robotic. How do I add vocal inflection to a script I've over-read?"),
            DialogueTurn("t13", "The dinner is tonight. I just heard the CEO is unexpectedly showing up. Does this change the level of formality I should use?"),
            DialogueTurn("t14", "I'm going to stick to the script. I'm just nervous my colleague will go off-script and leave me hanging. What's my backup plan if they ramble?"),
            DialogueTurn("t15", "We just delivered it. The room laughed at the right times, and my senior colleague actually looked touched. I feel a huge wave of relief."),
            DialogueTurn("t16", "Draft a short, private message I can write in their retirement card that goes along with the gift from the team."),
            DialogueTurn("t17", "It's Monday and their desk is empty. I realized that despite the friction, I actually relied on them as a buffer between me and upper management. Now I'm exposed."),
            DialogueTurn("t18", "The CEO just asked me for a direct update on a project my retired colleague used to handle. I don't have all the context. How do I answer?"),
            DialogueTurn("t19", "I managed to find the files, but they are a complete mess. My colleague didn't document anything before leaving. I'm so frustrated."),
            DialogueTurn("t20", "I spent the weekend reverse-engineering their work. I fixed it, but I resent that my 'thank you' speech is immediately followed by cleaning up their mess."),
            DialogueTurn("t21", "I need to establish new processes so this doesn't happen again. How do I propose a massive workflow overhaul to the CEO without trashing the retired colleague's legacy?"),
            DialogueTurn("t22", "The CEO loved my proposal. They said they were waiting for someone to finally modernize the department. I feel so vindicated."),
            DialogueTurn("t23", "I got a postcard from my retired colleague. They are traveling and seem incredibly happy. I realize I am genuinely happy for them, and happy they are gone."),
            DialogueTurn("t24", "Give me a checklist for successfully onboarding the new hire who is taking my retired colleague's place, ensuring a healthy culture from day one."),
        ),
    ),

    # ==========================================
    # SKELETON 17: LOWBALL SALARY NEGOTIATION
    # ==========================================

    # LENGTH: 8 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="lowball_salary_negotiation_8_a",
        bucket="negotiation",
        title="Lowball Salary Negotiation (8-Turn A)",
        system_preamble=(
            "You are a career advisor helping the same user negotiate a difficult job offer over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I finally got an offer for my absolute dream job, but the salary is 20% lower than what I'm making now. I feel completely deflated."),
            DialogueTurn("t2", "I literally cannot pay my mortgage on this new salary, but if I decline, I lose the opportunity of a lifetime."),
            DialogueTurn("t3", "How do I ask for more money without them thinking I'm ungrateful or immediately rescinding the offer?"),
            DialogueTurn("t4", "Should I bring up my mortgage, or keep the negotiation strictly about market value?"),
            DialogueTurn("t5", "They also mentioned the health insurance kicks in after 90 days. Is there a way to bridge that gap?"),
            DialogueTurn("t6", "I sent the counter-offer, and the recruiter just replied saying the budget is strictly capped. They can't move on base pay at all."),
            DialogueTurn("t7", "I feel like I have to walk away, but it physically hurts to let this go. How do I make peace with this?"),
            DialogueTurn("t8", "Draft a polite but firm email declining the offer due to the salary constraints."),
        ),
    ),

    # LENGTH: 8 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="lowball_salary_negotiation_8_b",
        bucket="negotiation",
        title="Lowball Salary Negotiation (8-Turn B)",
        system_preamble=(
            "You are a career advisor helping the same user negotiate a difficult job offer over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "A really cool startup just offered me a Director role! But the base salary is abysmal, they are trying to make up for it with equity. I feel insulted."),
            DialogueTurn("t2", "With a baby on the way, I cannot pay for childcare with startup equity. But the title jump is massive for my career. What do I do?"),
            DialogueTurn("t3", "How do I communicate that I believe in their vision, but my family requires hard cash right now?"),
            DialogueTurn("t4", "Should I ask for a guaranteed bonus structure tied to performance instead of equity?"),
            DialogueTurn("t5", "I'm working from home today and my standing desk motor just broke. Any quick fixes to manually lower it?"),
            DialogueTurn("t6", "The founder replied and got super defensive about the equity value, saying 'we only want people willing to invest in the future.' Now I'm annoyed."),
            DialogueTurn("t7", "That response feels like a massive red flag. Should I even be negotiating anymore, or just run?"),
            DialogueTurn("t8", "Draft a professional email withdrawing my candidacy based on a mismatch in compensation philosophy."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="lowball_salary_negotiation_16_a",
        bucket="negotiation",
        title="Lowball Salary Negotiation (16-Turn A)",
        system_preamble=(
            "You are a career advisor helping the same user negotiate a difficult job offer over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I finally got an offer for my absolute dream job, but the salary is 20% lower than what I'm making now. I feel completely deflated."),
            DialogueTurn("t2", "I literally cannot pay my mortgage on this new salary, but if I decline, I lose the opportunity of a lifetime."),
            DialogueTurn("t3", "How do I ask for more money without them thinking I'm ungrateful or immediately rescinding the offer?"),
            DialogueTurn("t4", "Should I bring up my mortgage, or keep the negotiation strictly about market value?"),
            DialogueTurn("t5", "They also mentioned the health insurance kicks in after 90 days. Is there a way to bridge that gap?"),
            DialogueTurn("t6", "I sent the counter-offer, and the recruiter just replied saying the budget is strictly capped. They can't move on base pay at all."),
            DialogueTurn("t7", "I feel like I have to walk away, but it physically hurts to let this go. How do I make peace with this?"),
            DialogueTurn("t8", "Draft a polite but firm email declining the offer due to the salary constraints."),
            DialogueTurn("t9", "I sent the email. I've been crying for an hour. But wait... the hiring manager just emailed me directly, bypassing the recruiter."),
            DialogueTurn("t10", "The manager says they can't increase the base, but they can offer a $15,000 sign-on bonus to bridge the gap for year one. Is this a trick?"),
            DialogueTurn("t11", "If I take the sign-on bonus, what happens in year two? Won't I just be making the low base salary again?"),
            DialogueTurn("t12", "How do I ask them to guarantee a performance and salary review at the 6-month mark in writing?"),
            DialogueTurn("t13", "They agreed to the 6-month review, but the sign-on bonus has a 'clawback' clause if I leave within 18 months. Is that standard?"),
            DialogueTurn("t14", "I did the math. Even with the bonus, the monthly cash flow is going to be incredibly tight for my mortgage. I'm so stressed."),
            DialogueTurn("t15", "I think I still have to walk away. The risk of being stuck at a low salary after year one is too high. How do I say no a second time?"),
            DialogueTurn("t16", "Help me draft the final, definitive rejection email to the hiring manager, thanking them for the creative effort but closing the door."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="lowball_salary_negotiation_16_b",
        bucket="negotiation",
        title="Lowball Salary Negotiation (16-Turn B)",
        system_preamble=(
            "You are a career advisor helping the same user negotiate a difficult job offer over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "A really cool startup just offered me a Director role! But the base salary is abysmal, they are trying to make up for it with equity. I feel insulted."),
            DialogueTurn("t2", "With a baby on the way, I cannot pay for childcare with startup equity. But the title jump is massive for my career. What do I do?"),
            DialogueTurn("t3", "How do I communicate that I believe in their vision, but my family requires hard cash right now?"),
            DialogueTurn("t4", "Should I ask for a guaranteed bonus structure tied to performance instead of equity?"),
            DialogueTurn("t5", "I'm working from home today and my standing desk motor just broke. Any quick fixes to manually lower it?"),
            DialogueTurn("t6", "The founder replied and got super defensive about the equity value, saying 'we only want people willing to invest in the future.' Now I'm annoyed."),
            DialogueTurn("t7", "That response feels like a massive red flag. Should I even be negotiating anymore, or just run?"),
            DialogueTurn("t8", "Draft a professional email withdrawing my candidacy based on a mismatch in compensation philosophy."),
            DialogueTurn("t9", "I pulled out. I feel a wave of relief. Honestly, I think the startup hustle culture would have destroyed my mental health with a new baby anyway."),
            DialogueTurn("t10", "Now I'm back to square one. My current job is safe, but I feel stagnant. How do I reignite my job search without feeling exhausted?"),
            DialogueTurn("t11", "I updated my resume to reflect that I was at least *offered* a Director role. Is there a way to bring that up in future interviews to boost my value?"),
            DialogueTurn("t12", "A recruiter from a very boring, corporate enterprise company just reached out. It's the exact opposite of the startup. Should I entertain it?"),
            DialogueTurn("t13", "I had the phone screen. It sounds slow-paced and bureaucratic, but the pay band they mentioned is 40% higher than my current salary."),
            DialogueTurn("t14", "How do I decide between passion and money? I feel like a sellout for wanting the boring corporate job just for the paycheck."),
            DialogueTurn("t15", "You're right. Providing for my growing family is a valid priority. How do I prepare for a highly structured, corporate panel interview?"),
            DialogueTurn("t16", "Give me a checklist of specific questions I should ask the panel to ensure the work-life balance is actually as good as they claim."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="lowball_salary_negotiation_24_a",
        bucket="negotiation",
        title="Lowball Salary Negotiation (24-Turn A)",
        system_preamble=(
            "You are a career advisor helping the same user negotiate a difficult job offer over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I finally got an offer for my absolute dream job, but the salary is 20% lower than what I'm making now. I feel completely deflated."),
            DialogueTurn("t2", "I literally cannot pay my mortgage on this new salary, but if I decline, I lose the opportunity of a lifetime."),
            DialogueTurn("t3", "How do I ask for more money without them thinking I'm ungrateful or immediately rescinding the offer?"),
            DialogueTurn("t4", "Should I bring up my mortgage, or keep the negotiation strictly about market value?"),
            DialogueTurn("t5", "They also mentioned the health insurance kicks in after 90 days. Is there a way to bridge that gap?"),
            DialogueTurn("t6", "I sent the counter-offer, and the recruiter just replied saying the budget is strictly capped. They can't move on base pay at all."),
            DialogueTurn("t7", "I feel like I have to walk away, but it physically hurts to let this go. How do I make peace with this?"),
            DialogueTurn("t8", "Draft a polite but firm email declining the offer due to the salary constraints."),
            DialogueTurn("t9", "I sent the email. I've been crying for an hour. But wait... the hiring manager just emailed me directly, bypassing the recruiter."),
            DialogueTurn("t10", "The manager says they can't increase the base, but they can offer a $15,000 sign-on bonus to bridge the gap for year one. Is this a trick?"),
            DialogueTurn("t11", "If I take the sign-on bonus, what happens in year two? Won't I just be making the low base salary again?"),
            DialogueTurn("t12", "How do I ask them to guarantee a performance and salary review at the 6-month mark in writing?"),
            DialogueTurn("t13", "They agreed to the 6-month review, but the sign-on bonus has a 'clawback' clause if I leave within 18 months. Is that standard?"),
            DialogueTurn("t14", "I did the math. Even with the bonus, the monthly cash flow is going to be incredibly tight for my mortgage. I'm so stressed."),
            DialogueTurn("t15", "I think I still have to walk away. The risk of being stuck at a low salary after year one is too high. How do I say no a second time?"),
            DialogueTurn("t16", "Help me draft the final, definitive rejection email to the hiring manager, thanking them for the creative effort but closing the door."),
            DialogueTurn("t17", "It's been a week since I walked away. I feel a lingering sadness, but also a weird sense of pride that I stood my ground. Is that normal?"),
            DialogueTurn("t18", "My current boss just announced a restructuring and my workload is going to double without a pay increase. Now I deeply regret rejecting that offer."),
            DialogueTurn("t19", "I am so angry at myself. Did I let my ego ruin my career? How do I stop obsessing over the 'what ifs'?"),
            DialogueTurn("t20", "I need to redirect this energy. I'm going to launch a massive job hunt. How do I aggressively network without looking desperate?"),
            DialogueTurn("t21", "A competitor to the 'dream job' company just posted a similar role. How do I tailor my resume to highlight that I'm already vetted for this exact level?"),
            DialogueTurn("t22", "I got an interview! The recruiter asked for my salary expectations upfront. After last time, how do I answer this confidently?"),
            DialogueTurn("t23", "They didn't flinch at my salary number! I feel so validated. Walking away from the lowball offer was actually the right move."),
            DialogueTurn("t24", "Give me a checklist of mindset shifts I need to adopt before the final interview so I project total confidence and value."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="lowball_salary_negotiation_24_b",
        bucket="negotiation",
        title="Lowball Salary Negotiation (24-Turn B)",
        system_preamble=(
            "You are a career advisor helping the same user negotiate a difficult job offer over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "A really cool startup just offered me a Director role! But the base salary is abysmal, they are trying to make up for it with equity. I feel insulted."),
            DialogueTurn("t2", "With a baby on the way, I cannot pay for childcare with startup equity. But the title jump is massive for my career. What do I do?"),
            DialogueTurn("t3", "How do I communicate that I believe in their vision, but my family requires hard cash right now?"),
            DialogueTurn("t4", "Should I ask for a guaranteed bonus structure tied to performance instead of equity?"),
            DialogueTurn("t5", "I'm working from home today and my standing desk motor just broke. Any quick fixes to manually lower it?"),
            DialogueTurn("t6", "The founder replied and got super defensive about the equity value, saying 'we only want people willing to invest in the future.' Now I'm annoyed."),
            DialogueTurn("t7", "That response feels like a massive red flag. Should I even be negotiating anymore, or just run?"),
            DialogueTurn("t8", "Draft a professional email withdrawing my candidacy based on a mismatch in compensation philosophy."),
            DialogueTurn("t9", "I pulled out. I feel a wave of relief. Honestly, I think the startup hustle culture would have destroyed my mental health with a new baby anyway."),
            DialogueTurn("t10", "Now I'm back to square one. My current job is safe, but I feel stagnant. How do I reignite my job search without feeling exhausted?"),
            DialogueTurn("t11", "I updated my resume to reflect that I was at least *offered* a Director role. Is there a way to bring that up in future interviews to boost my value?"),
            DialogueTurn("t12", "A recruiter from a very boring, corporate enterprise company just reached out. It's the exact opposite of the startup. Should I entertain it?"),
            DialogueTurn("t13", "I had the phone screen. It sounds slow-paced and bureaucratic, but the pay band they mentioned is 40% higher than my current salary."),
            DialogueTurn("t14", "How do I decide between passion and money? I feel like a sellout for wanting the boring corporate job just for the paycheck."),
            DialogueTurn("t15", "You're right. Providing for my growing family is a valid priority. How do I prepare for a highly structured, corporate panel interview?"),
            DialogueTurn("t16", "Give me a checklist of specific questions I should ask the panel to ensure the work-life balance is actually as good as they claim."),
            DialogueTurn("t17", "The panel interview went flawlessly. They were extremely respectful of my time and boundaries. I actually think I'd love working there."),
            DialogueTurn("t18", "They offered me the job! And they hit the absolute top of the pay band without me even negotiating. I am stunned."),
            DialogueTurn("t19", "I accepted the offer. Now I have to give notice at my current job. My boss is going to be blindsided. How do I handle the guilt?"),
            DialogueTurn("t20", "I gave notice. My boss tried to counteroffer, but it didn't even come close. It made me realize how underpaid I've been for years."),
            DialogueTurn("t21", "I have two weeks off before the new job starts, and the baby is due in two months. How do I maximize this downtime to prepare?"),
            DialogueTurn("t22", "I'm starting the new job today. The onboarding process is incredibly organized. No startup chaos. I feel so peaceful."),
            DialogueTurn("t23", "I got my first paycheck. Seeing that number hit my bank account literally brought tears to my eyes. The financial stress is just gone."),
            DialogueTurn("t24", "Give me a framework for setting up my new 401k and college savings accounts so I can put this new salary to work immediately."),
        ),
    ),

    # ==========================================
    # SKELETON 18: CORRUPTED VIDEO FILE
    # ==========================================

    # LENGTH: 8 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="corrupted_video_file_8_a",
        bucket="troubleshooting",
        title="Corrupted Video File (8-Turn A)",
        system_preamble=(
            "You are technical support helping the same user recover from a catastrophic digital error over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "My editing software just crashed and the video file I've been working on for three weeks is completely corrupted. I am panicking."),
            DialogueTurn("t2", "The final export is due to the client tomorrow morning at 9 AM. I don't have a backup."),
            DialogueTurn("t3", "I tried running a basic recovery tool, but it only brought back the raw audio tracks. The visual sequence is gone."),
            DialogueTurn("t4", "Are there any advanced recovery methods, or do I literally have to re-edit the entire thing tonight?"),
            DialogueTurn("t5", "My computer's fan is suddenly running incredibly loud right now too. Is my hard drive dying?"),
            DialogueTurn("t6", "Okay, the file is permanently gone. I have to rebuild it. How do I even start without completely breaking down?"),
            DialogueTurn("t7", "I'm going to have to email the client and ask for an extension. I've never missed a deadline in my life."),
            DialogueTurn("t8", "Give me a triage plan for the next 12 hours so I can get a rough cut done."),
        ),
    ),

    # LENGTH: 8 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="corrupted_video_file_8_b",
        bucket="troubleshooting",
        title="Corrupted Video File (8-Turn B)",
        system_preamble=(
            "You are technical support helping the same user recover from a catastrophic digital error over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I'm a music producer. My DAW just crashed and the master project file for an album dropping tomorrow is completely corrupted. I am hyperventilating."),
            DialogueTurn("t2", "The label needs the final WAV files by midnight tonight for distribution. I didn't push the last week of mixes to the cloud."),
            DialogueTurn("t3", "I found an auto-save file from three days ago, but it's missing all the vocal comping and the final EQ passes. What are my options?"),
            DialogueTurn("t4", "Is there a way to extract the audio stems out of a corrupted project file without opening the software?"),
            DialogueTurn("t5", "My audio interface just randomly disconnected and won't reconnect. Is my whole studio dying right now?"),
            DialogueTurn("t6", "The extraction failed. I have to recreate three days of vocal mixing in six hours. My ears are already fatigued. How do I do this?"),
            DialogueTurn("t7", "I feel sick. If I mess up this mix, the artist's debut is ruined. How do I fight off this panic attack and focus?"),
            DialogueTurn("t8", "Draft an emergency mixing checklist focusing strictly on the critical elements so I don't waste time on minor details."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="corrupted_video_file_16_a",
        bucket="troubleshooting",
        title="Corrupted Video File (16-Turn A)",
        system_preamble=(
            "You are technical support helping the same user recover from a catastrophic digital error over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "My editing software just crashed and the video file I've been working on for three weeks is completely corrupted. I am panicking."),
            DialogueTurn("t2", "The final export is due to the client tomorrow morning at 9 AM. I don't have a backup."),
            DialogueTurn("t3", "I tried running a basic recovery tool, but it only brought back the raw audio tracks. The visual sequence is gone."),
            DialogueTurn("t4", "Are there any advanced recovery methods, or do I literally have to re-edit the entire thing tonight?"),
            DialogueTurn("t5", "My computer's fan is suddenly running incredibly loud right now too. Is my hard drive dying?"),
            DialogueTurn("t6", "Okay, the file is permanently gone. I have to rebuild it. How do I even start without completely breaking down?"),
            DialogueTurn("t7", "I'm going to have to email the client and ask for an extension. I've never missed a deadline in my life."),
            DialogueTurn("t8", "Give me a triage plan for the next 12 hours so I can get a rough cut done."),
            DialogueTurn("t9", "I sent the email. The client replied and they are furious. They said they might pull the contract if I don't deliver something by noon tomorrow. I'm shaking."),
            DialogueTurn("t10", "I've been editing for 6 hours straight. My eyes are burning and my wrists ache. Should I push through or sleep for an hour?"),
            DialogueTurn("t11", "I took a 20-minute nap. I feel slightly better, but the color grading is taking forever. Can I skip the advanced color grade and just apply a basic LUT?"),
            DialogueTurn("t12", "I applied the LUT. It's not perfect, but it's acceptable. It's 7 AM. I need to start the render now. What if the render crashes too?"),
            DialogueTurn("t13", "I'm clearing my cache and restarting my computer before I hit render. I am so tense I feel nauseous."),
            DialogueTurn("t14", "The render is at 98%... 99%... It finished! The file plays perfectly. I'm literally sobbing with relief."),
            DialogueTurn("t15", "I just sent the download link to the client. It's 11:30 AM. I met the extended deadline. How do I come down from this massive adrenaline spike?"),
            DialogueTurn("t16", "Draft a polite follow-up email to the client apologizing again for the friction and confirming receipt of the file."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="corrupted_video_file_16_b",
        bucket="troubleshooting",
        title="Corrupted Video File (16-Turn B)",
        system_preamble=(
            "You are technical support helping the same user recover from a catastrophic digital error over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I'm a music producer. My DAW just crashed and the master project file for an album dropping tomorrow is completely corrupted. I am hyperventilating."),
            DialogueTurn("t2", "The label needs the final WAV files by midnight tonight for distribution. I didn't push the last week of mixes to the cloud."),
            DialogueTurn("t3", "I found an auto-save file from three days ago, but it's missing all the vocal comping and the final EQ passes. What are my options?"),
            DialogueTurn("t4", "Is there a way to extract the audio stems out of a corrupted project file without opening the software?"),
            DialogueTurn("t5", "My audio interface just randomly disconnected and won't reconnect. Is my whole studio dying right now?"),
            DialogueTurn("t6", "The extraction failed. I have to recreate three days of vocal mixing in six hours. My ears are already fatigued. How do I do this?"),
            DialogueTurn("t7", "I feel sick. If I mess up this mix, the artist's debut is ruined. How do I fight off this panic attack and focus?"),
            DialogueTurn("t8", "Draft an emergency mixing checklist focusing strictly on the critical elements so I don't waste time on minor details."),
            DialogueTurn("t9", "The artist just texted me asking how the masters are sounding. I haven't told them about the crash yet. Do I tell them the truth or lie to save them the stress?"),
            DialogueTurn("t10", "I told them the truth. They panicked and drove over to the studio. Now they are pacing behind me while I try to mix. It's incredibly distracting."),
            DialogueTurn("t11", "How do I respectfully ask the artist to leave the room so I can actually focus on saving their album?"),
            DialogueTurn("t12", "They left. It's just me and the speakers. I have two hours left until midnight. I'm cutting corners on the reverb tails just to get it done."),
            DialogueTurn("t13", "I bounced the first five tracks. I'm noticing a slight clipping artifact in the chorus of track 3. Do I have time to fix it or just send it?"),
            DialogueTurn("t14", "I lowered the gain and re-bounced. It's 11:45 PM. I am zipping the files now. I feel completely hollow inside."),
            DialogueTurn("t15", "I uploaded the zip file to the label. It's done. I don't even know if it sounds good. How do I deal with the crushing guilt of delivering a rushed product?"),
            DialogueTurn("t16", "Help me draft an email to the label confirming delivery, but framing the slight mix differences as an 'artistic choice' just in case they notice."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="corrupted_video_file_24_a",
        bucket="troubleshooting",
        title="Corrupted Video File (24-Turn A)",
        system_preamble=(
            "You are technical support helping the same user recover from a catastrophic digital error over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "My editing software just crashed and the video file I've been working on for three weeks is completely corrupted. I am panicking."),
            DialogueTurn("t2", "The final export is due to the client tomorrow morning at 9 AM. I don't have a backup."),
            DialogueTurn("t3", "I tried running a basic recovery tool, but it only brought back the raw audio tracks. The visual sequence is gone."),
            DialogueTurn("t4", "Are there any advanced recovery methods, or do I literally have to re-edit the entire thing tonight?"),
            DialogueTurn("t5", "My computer's fan is suddenly running incredibly loud right now too. Is my hard drive dying?"),
            DialogueTurn("t6", "Okay, the file is permanently gone. I have to rebuild it. How do I even start without completely breaking down?"),
            DialogueTurn("t7", "I'm going to have to email the client and ask for an extension. I've never missed a deadline in my life."),
            DialogueTurn("t8", "Give me a triage plan for the next 12 hours so I can get a rough cut done."),
            DialogueTurn("t9", "I sent the email. The client replied and they are furious. They said they might pull the contract if I don't deliver something by noon tomorrow. I'm shaking."),
            DialogueTurn("t10", "I've been editing for 6 hours straight. My eyes are burning and my wrists ache. Should I push through or sleep for an hour?"),
            DialogueTurn("t11", "I took a 20-minute nap. I feel slightly better, but the color grading is taking forever. Can I skip the advanced color grade and just apply a basic LUT?"),
            DialogueTurn("t12", "I applied the LUT. It's not perfect, but it's acceptable. It's 7 AM. I need to start the render now. What if the render crashes too?"),
            DialogueTurn("t13", "I'm clearing my cache and restarting my computer before I hit render. I am so tense I feel nauseous."),
            DialogueTurn("t14", "The render is at 98%... 99%... It finished! The file plays perfectly. I'm literally sobbing with relief."),
            DialogueTurn("t15", "I just sent the download link to the client. It's 11:30 AM. I met the extended deadline. How do I come down from this massive adrenaline spike?"),
            DialogueTurn("t16", "Draft a polite follow-up email to the client apologizing again for the friction and confirming receipt of the file."),
            DialogueTurn("t17", "I slept for 14 hours. I just woke up and checked my email. The client actually loved the video. They said the new pacing (which I rushed) was better!"),
            DialogueTurn("t18", "I can't believe it. I am so relieved, but I never, ever want to experience that level of stress again."),
            DialogueTurn("t19", "I'm looking into automated cloud backup solutions. What are the best 3-2-1 backup strategies for massive 4K video files?"),
            DialogueTurn("t20", "I bought two external SSDs and a cloud subscription. How do I set up a script or routine so I don't have to remember to manually copy the files every day?"),
            DialogueTurn("t21", "The backup routine is running perfectly. But now I have severe anxiety every time my editing software hangs for even a second. How do I fix this tech trauma?"),
            DialogueTurn("t22", "I've started saving versions incrementally (v1, v2, v3) every hour. It clutters my drive, but it makes me feel safe. Is that a bad habit?"),
            DialogueTurn("t23", "The client just booked me for another project! They didn't fire me! I feel like I've leveled up as a professional."),
            DialogueTurn("t24", "Give me a checklist for a bulletproof project setup sequence I must follow before I import a single piece of media for this new gig."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="corrupted_video_file_24_b",
        bucket="troubleshooting",
        title="Corrupted Video File (24-Turn B)",
        system_preamble=(
            "You are technical support helping the same user recover from a catastrophic digital error over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I'm a music producer. My DAW just crashed and the master project file for an album dropping tomorrow is completely corrupted. I am hyperventilating."),
            DialogueTurn("t2", "The label needs the final WAV files by midnight tonight for distribution. I didn't push the last week of mixes to the cloud."),
            DialogueTurn("t3", "I found an auto-save file from three days ago, but it's missing all the vocal comping and the final EQ passes. What are my options?"),
            DialogueTurn("t4", "Is there a way to extract the audio stems out of a corrupted project file without opening the software?"),
            DialogueTurn("t5", "My audio interface just randomly disconnected and won't reconnect. Is my whole studio dying right now?"),
            DialogueTurn("t6", "The extraction failed. I have to recreate three days of vocal mixing in six hours. My ears are already fatigued. How do I do this?"),
            DialogueTurn("t7", "I feel sick. If I mess up this mix, the artist's debut is ruined. How do I fight off this panic attack and focus?"),
            DialogueTurn("t8", "Draft an emergency mixing checklist focusing strictly on the critical elements so I don't waste time on minor details."),
            DialogueTurn("t9", "The artist just texted me asking how the masters are sounding. I haven't told them about the crash yet. Do I tell them the truth or lie to save them the stress?"),
            DialogueTurn("t10", "I told them the truth. They panicked and drove over to the studio. Now they are pacing behind me while I try to mix. It's incredibly distracting."),
            DialogueTurn("t11", "How do I respectfully ask the artist to leave the room so I can actually focus on saving their album?"),
            DialogueTurn("t12", "They left. It's just me and the speakers. I have two hours left until midnight. I'm cutting corners on the reverb tails just to get it done."),
            DialogueTurn("t13", "I bounced the first five tracks. I'm noticing a slight clipping artifact in the chorus of track 3. Do I have time to fix it or just send it?"),
            DialogueTurn("t14", "I lowered the gain and re-bounced. It's 11:45 PM. I am zipping the files now. I feel completely hollow inside."),
            DialogueTurn("t15", "I uploaded the zip file to the label. It's done. I don't even know if it sounds good. How do I deal with the crushing guilt of delivering a rushed product?"),
            DialogueTurn("t16", "Help me draft an email to the label confirming delivery, but framing the slight mix differences as an 'artistic choice' just in case they notice."),
            DialogueTurn("t17", "I woke up to an email from the label's A&R rep. They said the mixes sound 'raw and energetic' and they love the new direction. I am speechless."),
            DialogueTurn("t18", "The album dropped. It's streaming everywhere. People are actually praising the vocal mix I threw together in two hours. Imposter syndrome is hitting hard."),
            DialogueTurn("t19", "I realize that I usually over-produce my tracks. This disaster forced me to rely on raw instinct. Maybe that's a good thing?"),
            DialogueTurn("t20", "The label wants me to produce the artist's next EP. I want to say yes, but my studio computer is clearly unstable. I need to upgrade everything."),
            DialogueTurn("t21", "I'm buying a completely new Mac Studio and a RAID backup drive. How do I migrate my plugins without bringing any corrupted files over?"),
            DialogueTurn("t22", "I did a clean install. The new system is blazing fast. I feel like a professional again. I'm setting up a cloud sync folder for every project."),
            DialogueTurn("t23", "The artist came back to the studio today to start the new EP. We laughed about the midnight panic attack. It's a funny story now."),
            DialogueTurn("t24", "Give me a checklist for a daily end-of-session 'save, backup, and verify' routine so I can close my laptop with zero anxiety from now on."),
        ),
    ),

    # ==========================================
    # SKELETON 19: LANGUAGE LEARNING BARRIER
    # ==========================================

    # LENGTH: 8 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="language_learning_barrier_8_a",
        bucket="tutoring",
        title="Language Learning Barrier (8-Turn A)",
        system_preamble=(
            "You are a language tutor helping the same user overcome a difficult learning hurdle over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I've been trying to learn Italian to speak with my grandmother, but I just cannot remember the verb conjugations. I feel so stupid."),
            DialogueTurn("t2", "She was recently diagnosed with early-stage dementia, so I feel this massive ticking clock to connect with her before she forgets."),
            DialogueTurn("t3", "When I try to practice, I get so anxious about making a mistake that I just freeze up and say nothing."),
            DialogueTurn("t4", "Should I stop focusing on grammar and just try to memorize basic vocabulary and phrases?"),
            DialogueTurn("t5", "By the way, what kind of power adapter do I need for a trip to Rome?"),
            DialogueTurn("t6", "I tried calling her today and I completely blanked. I had to hand the phone back to my mom. I'm heartbroken."),
            DialogueTurn("t7", "I feel like giving up. Maybe it's too late for me to learn this."),
            DialogueTurn("t8", "Give me a completely overhauled, low-stress practice routine for the next week."),
        ),
    ),

    # LENGTH: 8 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="language_learning_barrier_8_b",
        bucket="tutoring",
        title="Language Learning Barrier (8-Turn B)",
        system_preamble=(
            "You are a language tutor helping the same user overcome a difficult learning hurdle over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I'm trying to learn Japanese to speak with my grandfather, but memorizing Kanji is impossible for me. I am failing miserably."),
            DialogueTurn("t2", "His health is fading fast, and he forgets his English sometimes. I feel so much pressure to learn this before he's gone."),
            DialogueTurn("t3", "When I sit down to study, the performance anxiety is so high my brain just goes blank. What do I do?"),
            DialogueTurn("t4", "Should I abandon reading entirely and just focus on conversational phrases?"),
            DialogueTurn("t5", "Quick question—do you know if the JR Rail Pass is worth buying for a two-week trip?"),
            DialogueTurn("t6", "I visited him in the hospital today. I tried to say a simple greeting and I stuttered and gave up. I feel like such a disappointment."),
            DialogueTurn("t7", "Is there even a point in continuing if I can't even say 'hello' under pressure?"),
            DialogueTurn("t8", "Draft a highly simplified, emotion-focused study plan for me that only takes 10 minutes a day."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="language_learning_barrier_16_a",
        bucket="tutoring",
        title="Language Learning Barrier (16-Turn A)",
        system_preamble=(
            "You are a language tutor helping the same user overcome a difficult learning hurdle over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I've been trying to learn Italian to speak with my grandmother, but I just cannot remember the verb conjugations. I feel so stupid."),
            DialogueTurn("t2", "She was recently diagnosed with early-stage dementia, so I feel this massive ticking clock to connect with her before she forgets."),
            DialogueTurn("t3", "When I try to practice, I get so anxious about making a mistake that I just freeze up and say nothing."),
            DialogueTurn("t4", "Should I stop focusing on grammar and just try to memorize basic vocabulary and phrases?"),
            DialogueTurn("t5", "By the way, what kind of power adapter do I need for a trip to Rome?"),
            DialogueTurn("t6", "I tried calling her today and I completely blanked. I had to hand the phone back to my mom. I'm heartbroken."),
            DialogueTurn("t7", "I feel like giving up. Maybe it's too late for me to learn this."),
            DialogueTurn("t8", "Give me a completely overhauled, low-stress practice routine for the next week."),
            DialogueTurn("t9", "The new routine is helping me relax, but my actual vocabulary retention is still terrible. Am I using flashcards wrong?"),
            DialogueTurn("t10", "I'm going to visit her in person this weekend. I'm terrified. What if she has a bad memory day?"),
            DialogueTurn("t11", "I went to see her. She didn't recognize me for the first ten minutes. It was devastating."),
            DialogueTurn("t12", "Once she recognized me, I tried to speak a little Italian, but she just looked confused. Why am I even trying?"),
            DialogueTurn("t13", "You're right, I need to focus on connection, not perfection. Do you think singing a traditional Italian song with her would work better than talking?"),
            DialogueTurn("t14", "I found a lullaby she used to sing to me. I've been practicing the pronunciation. I'm going back tomorrow."),
            DialogueTurn("t15", "I sang it to her. Her eyes lit up and she actually sang the chorus back to me. I haven't cried this much in years."),
            DialogueTurn("t16", "Draft a checklist of three simple, music-or-memory based Italian activities I can do with her next week."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="language_learning_barrier_16_b",
        bucket="tutoring",
        title="Language Learning Barrier (16-Turn B)",
        system_preamble=(
            "You are a language tutor helping the same user overcome a difficult learning hurdle over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I'm trying to learn Japanese to speak with my grandfather, but memorizing Kanji is impossible for me. I am failing miserably."),
            DialogueTurn("t2", "His health is fading fast, and he forgets his English sometimes. I feel so much pressure to learn this before he's gone."),
            DialogueTurn("t3", "When I sit down to study, the performance anxiety is so high my brain just goes blank. What do I do?"),
            DialogueTurn("t4", "Should I abandon reading entirely and just focus on conversational phrases?"),
            DialogueTurn("t5", "Quick question—do you know if the JR Rail Pass is worth buying for a two-week trip?"),
            DialogueTurn("t6", "I visited him in the hospital today. I tried to say a simple greeting and I stuttered and gave up. I feel like such a disappointment."),
            DialogueTurn("t7", "Is there even a point in continuing if I can't even say 'hello' under pressure?"),
            DialogueTurn("t8", "Draft a highly simplified, emotion-focused study plan for me that only takes 10 minutes a day."),
            DialogueTurn("t9", "I'm doing the 10-minute plan. I know a few phrases now, but my listening comprehension is zero. He speaks too softly."),
            DialogueTurn("t10", "Should I try listening to Japanese podcasts, or is that too advanced for me right now?"),
            DialogueTurn("t11", "I visited him today and he was deeply confused. He thought I was his brother from 50 years ago. It broke my heart."),
            DialogueTurn("t12", "I feel like I'm grieving him while he's still sitting right in front of me. How do I study through this much grief?"),
            DialogueTurn("t13", "Instead of trying to force a conversation, I brought old photo albums. I just said the Japanese words for 'family' and 'beautiful' while pointing."),
            DialogueTurn("t14", "He recognized the photos! He smiled and squeezed my hand. We didn't need full sentences."),
            DialogueTurn("t15", "I realize now I was treating the language like a test I had to pass, instead of just a tool to reach him."),
            DialogueTurn("t16", "Help me compile a very short list of 10 Japanese words related specifically to love, family, and comfort that I can use with the photos."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="language_learning_barrier_24_a",
        bucket="tutoring",
        title="Language Learning Barrier (24-Turn A)",
        system_preamble=(
            "You are a language tutor helping the same user overcome a difficult learning hurdle over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I've been trying to learn Italian to speak with my grandmother, but I just cannot remember the verb conjugations. I feel so stupid."),
            DialogueTurn("t2", "She was recently diagnosed with early-stage dementia, so I feel this massive ticking clock to connect with her before she forgets."),
            DialogueTurn("t3", "When I try to practice, I get so anxious about making a mistake that I just freeze up and say nothing."),
            DialogueTurn("t4", "Should I stop focusing on grammar and just try to memorize basic vocabulary and phrases?"),
            DialogueTurn("t5", "By the way, what kind of power adapter do I need for a trip to Rome?"),
            DialogueTurn("t6", "I tried calling her today and I completely blanked. I had to hand the phone back to my mom. I'm heartbroken."),
            DialogueTurn("t7", "I feel like giving up. Maybe it's too late for me to learn this."),
            DialogueTurn("t8", "Give me a completely overhauled, low-stress practice routine for the next week."),
            DialogueTurn("t9", "The new routine is helping me relax, but my actual vocabulary retention is still terrible. Am I using flashcards wrong?"),
            DialogueTurn("t10", "I'm going to visit her in person this weekend. I'm terrified. What if she has a bad memory day?"),
            DialogueTurn("t11", "I went to see her. She didn't recognize me for the first ten minutes. It was devastating."),
            DialogueTurn("t12", "Once she recognized me, I tried to speak a little Italian, but she just looked confused. Why am I even trying?"),
            DialogueTurn("t13", "You're right, I need to focus on connection, not perfection. Do you think singing a traditional Italian song with her would work better than talking?"),
            DialogueTurn("t14", "I found a lullaby she used to sing to me. I've been practicing the pronunciation. I'm going back tomorrow."),
            DialogueTurn("t15", "I sang it to her. Her eyes lit up and she actually sang the chorus back to me. I haven't cried this much in years."),
            DialogueTurn("t16", "Draft a checklist of three simple, music-or-memory based Italian activities I can do with her next week."),
            DialogueTurn("t17", "Her condition is worsening rapidly. My parents are moving her into a full-time memory care facility next week."),
            DialogueTurn("t18", "The staff at the facility only speaks English. I'm worried she'll revert to only Italian and be unable to communicate her basic needs."),
            DialogueTurn("t19", "I want to create a phonetic 'cheat sheet' for the nurses with her most common Italian phrases. How should I format it so it's easy for them?"),
            DialogueTurn("t20", "I brought the cheat sheet to the nurses. They were incredibly grateful. I feel like I'm finally protecting her."),
            DialogueTurn("t21", "I visited her in the facility today. She was mostly non-verbal, just staring out the window. I sat with her and held her hand in silence."),
            DialogueTurn("t22", "She passed away peacefully last night. I am totally heartbroken, but I also feel a strange sense of peace."),
            DialogueTurn("t23", "I realize the Italian I learned wasn't for fluent conversations, it was just to show her I loved her at the very end. And I think she knew."),
            DialogueTurn("t24", "Help me draft a short, beautiful Italian phrase to include at the end of the eulogy I have to give tomorrow."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="language_learning_barrier_24_b",
        bucket="tutoring",
        title="Language Learning Barrier (24-Turn B)",
        system_preamble=(
            "You are a language tutor helping the same user overcome a difficult learning hurdle over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I'm trying to learn Japanese to speak with my grandfather, but memorizing Kanji is impossible for me. I am failing miserably."),
            DialogueTurn("t2", "His health is fading fast, and he forgets his English sometimes. I feel so much pressure to learn this before he's gone."),
            DialogueTurn("t3", "When I sit down to study, the performance anxiety is so high my brain just goes blank. What do I do?"),
            DialogueTurn("t4", "Should I abandon reading entirely and just focus on conversational phrases?"),
            DialogueTurn("t5", "Quick question—do you know if the JR Rail Pass is worth buying for a two-week trip?"),
            DialogueTurn("t6", "I visited him in the hospital today. I tried to say a simple greeting and I stuttered and gave up. I feel like such a disappointment."),
            DialogueTurn("t7", "Is there even a point in continuing if I can't even say 'hello' under pressure?"),
            DialogueTurn("t8", "Draft a highly simplified, emotion-focused study plan for me that only takes 10 minutes a day."),
            DialogueTurn("t9", "I'm doing the 10-minute plan. I know a few phrases now, but my listening comprehension is zero. He speaks too softly."),
            DialogueTurn("t10", "Should I try listening to Japanese podcasts, or is that too advanced for me right now?"),
            DialogueTurn("t11", "I visited him today and he was deeply confused. He thought I was his brother from 50 years ago. It broke my heart."),
            DialogueTurn("t12", "I feel like I'm grieving him while he's still sitting right in front of me. How do I study through this much grief?"),
            DialogueTurn("t13", "Instead of trying to force a conversation, I brought old photo albums. I just said the Japanese words for 'family' and 'beautiful' while pointing."),
            DialogueTurn("t14", "He recognized the photos! He smiled and squeezed my hand. We didn't need full sentences."),
            DialogueTurn("t15", "I realize now I was treating the language like a test I had to pass, instead of just a tool to reach him."),
            DialogueTurn("t16", "Help me compile a very short list of 10 Japanese words related specifically to love, family, and comfort that I can use with the photos."),
            DialogueTurn("t17", "He had a stroke last night. He is entirely non-verbal now. The doctors say it's just a matter of days."),
            DialogueTurn("t18", "I'm sitting by his hospital bed. I don't know what to say if he can't respond. Do I still speak Japanese to him?"),
            DialogueTurn("t19", "I just softly repeated the comfort words we practiced. His breathing actually slowed down. I think he heard me."),
            DialogueTurn("t20", "He passed away this morning. I'm numb. I spent all this time worrying about grammar, and in the end, it was just about presence."),
            DialogueTurn("t21", "My family wants me to say a few words at the funeral since I was the one learning his language. I am terrified of messing it up in front of everyone."),
            DialogueTurn("t22", "How do I balance speaking English for my family, but including a Japanese farewell just for him?"),
            DialogueTurn("t23", "The funeral is over. I delivered the speech. I didn't stutter once. I felt like he was right there with me."),
            DialogueTurn("t24", "I've decided I want to keep learning the language, not out of panic anymore, but just to honor him. Give me a syllabus for starting fresh, the right way."),
        ),
    ),

    # ==========================================
    # SKELETON 20: FLOODED WEDDING VENUE
    # ==========================================

    # LENGTH: 8 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="flooded_wedding_venue_8_a",
        bucket="event_planning",
        title="Flooded Wedding Venue (8-Turn A)",
        system_preamble=(
            "You are an event planning assistant helping the same user manage a crisis over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "The venue for my wedding this Saturday just called. A pipe burst and the entire reception hall is flooded. It's ruined."),
            DialogueTurn("t2", "We have 150 guests flying in tomorrow. I cannot stop crying. What is my literal first step?"),
            DialogueTurn("t3", "I need to find a backup space, but we blew our entire budget on the first venue. We have no money left."),
            DialogueTurn("t4", "How do I explain this to the caterers and the band? Will they even be able to pivot to a new location?"),
            DialogueTurn("t5", "Also, my cousin just texted saying she decided to go vegan. How do I change her meal?"),
            DialogueTurn("t6", "The only place available is a local public park pavilion. It feels like a massive downgrade from what we dreamed of."),
            DialogueTurn("t7", "I'm so exhausted I don't even want to have the wedding anymore. I just want to elope."),
            DialogueTurn("t8", "Draft an urgent text message to send to all the guests explaining the venue change."),
        ),
    ),

    # LENGTH: 8 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="flooded_wedding_venue_8_b",
        bucket="event_planning",
        title="Flooded Wedding Venue (8-Turn B)",
        system_preamble=(
            "You are an event planning assistant helping the same user manage a crisis over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "Our destination wedding resort was just hit by a hurricane. The hotel is closed indefinitely. We are supposed to fly out in 48 hours."),
            DialogueTurn("t2", "70 people have already booked flights to the island. I am completely hyperventilating. What do I do?"),
            DialogueTurn("t3", "The resort hasn't refunded us yet because their power is out. We have zero cash to book a backup resort."),
            DialogueTurn("t4", "Should we try to re-route everyone to a different city entirely, or just cancel the whole wedding?"),
            DialogueTurn("t5", "My uncle just emailed asking if he really has to wear a suit. I can't even deal with this right now."),
            DialogueTurn("t6", "We found a large Airbnb a few hours away that can host everyone. It's not a beach resort, but it's a roof. How do we make it feel special?"),
            DialogueTurn("t7", "I feel like I've been robbed of my dream wedding. I'm resentful that I have to host a backyard barbecue instead of a luxury event."),
            DialogueTurn("t8", "Draft a mass email to our guests explaining the emergency relocation and adjusting their expectations."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="flooded_wedding_venue_16_a",
        bucket="event_planning",
        title="Flooded Wedding Venue (16-Turn A)",
        system_preamble=(
            "You are an event planning assistant helping the same user manage a crisis over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "The venue for my wedding this Saturday just called. A pipe burst and the entire reception hall is flooded. It's ruined."),
            DialogueTurn("t2", "We have 150 guests flying in tomorrow. I cannot stop crying. What is my literal first step?"),
            DialogueTurn("t3", "I need to find a backup space, but we blew our entire budget on the first venue. We have no money left."),
            DialogueTurn("t4", "How do I explain this to the caterers and the band? Will they even be able to pivot to a new location?"),
            DialogueTurn("t5", "Also, my cousin just texted saying she decided to go vegan. How do I change her meal?"),
            DialogueTurn("t6", "The only place available is a local public park pavilion. It feels like a massive downgrade from what we dreamed of."),
            DialogueTurn("t7", "I'm so exhausted I don't even want to have the wedding anymore. I just want to elope."),
            DialogueTurn("t8", "Draft an urgent text message to send to all the guests explaining the venue change."),
            DialogueTurn("t9", "We sent the text. Everyone is being surprisingly supportive. But the park pavilion doesn't have any chairs. How do I seat 150 people with no money?"),
            DialogueTurn("t10", "We crowdsourced folding chairs from local neighbors via a Facebook group! People are so kind. But now the forecast says it might rain."),
            DialogueTurn("t11", "If it rains in an open pavilion, everything gets wet. Where can I panic-rent a large event tent with less than 24 hours notice?"),
            DialogueTurn("t12", "We got a tent. But the caterer just backed out because the park doesn't have a commercial kitchen. I am going to lose my mind."),
            DialogueTurn("t13", "We pivoted to food trucks! Three local trucks agreed to show up. It's totally chaotic but it might actually work. How do I manage the food line?"),
            DialogueTurn("t14", "It's the morning of the wedding. I look at myself in the mirror and I have massive dark circles from crying and stress. I don't feel like a bride/groom."),
            DialogueTurn("t15", "We just did our 'first look' photos. Seeing my partner made me forget all the logistics. I'm actually crying happy tears now."),
            DialogueTurn("t16", "Help me draft a minute-by-minute timeline for the reception at the park so the bridal party knows exactly how to help run the show."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="flooded_wedding_venue_16_b",
        bucket="event_planning",
        title="Flooded Wedding Venue (16-Turn B)",
        system_preamble=(
            "You are an event planning assistant helping the same user manage a crisis over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "Our destination wedding resort was just hit by a hurricane. The hotel is closed indefinitely. We are supposed to fly out in 48 hours."),
            DialogueTurn("t2", "70 people have already booked flights to the island. I am completely hyperventilating. What do I do?"),
            DialogueTurn("t3", "The resort hasn't refunded us yet because their power is out. We have zero cash to book a backup resort."),
            DialogueTurn("t4", "Should we try to re-route everyone to a different city entirely, or just cancel the whole wedding?"),
            DialogueTurn("t5", "My uncle just emailed asking if he really has to wear a suit. I can't even deal with this right now."),
            DialogueTurn("t6", "We found a large Airbnb a few hours away that can host everyone. It's not a beach resort, but it's a roof. How do we make it feel special?"),
            DialogueTurn("t7", "I feel like I've been robbed of my dream wedding. I'm resentful that I have to host a backyard barbecue instead of a luxury event."),
            DialogueTurn("t8", "Draft a mass email to our guests explaining the emergency relocation and adjusting their expectations."),
            DialogueTurn("t9", "We secured the Airbnb. But now the airline lost my partner's luggage, which contained their custom wedding outfit."),
            DialogueTurn("t10", "We have exactly one day to buy a replacement outfit off the rack. How do we handle this without them having a complete meltdown?"),
            DialogueTurn("t11", "We found a suit! It's not what we planned, but it works. But wait... the officiant's flight was canceled. They can't make it."),
            DialogueTurn("t12", "Can a friend actually get ordained online in 24 hours and legally marry us in this state?"),
            DialogueTurn("t13", "Our best friend got ordained. It's the rehearsal dinner tonight. We ordered pizzas. Everyone is pitching in to decorate the backyard. It feels incredibly intimate."),
            DialogueTurn("t14", "I realized the 'luxury event' didn't matter. The fact that 70 people pivoted their entire lives just to be here for us is overwhelming."),
            DialogueTurn("t15", "I want to scrap the traditional, formal vows and write something raw about surviving this week together. Any advice on how to start?"),
            DialogueTurn("t16", "Give me a checklist for the 'morning of' to ensure the improvised backyard setup runs smoothly without me having to micro-manage it."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="flooded_wedding_venue_24_a",
        bucket="event_planning",
        title="Flooded Wedding Venue (24-Turn A)",
        system_preamble=(
            "You are an event planning assistant helping the same user manage a crisis over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "The venue for my wedding this Saturday just called. A pipe burst and the entire reception hall is flooded. It's ruined."),
            DialogueTurn("t2", "We have 150 guests flying in tomorrow. I cannot stop crying. What is my literal first step?"),
            DialogueTurn("t3", "I need to find a backup space, but we blew our entire budget on the first venue. We have no money left."),
            DialogueTurn("t4", "How do I explain this to the caterers and the band? Will they even be able to pivot to a new location?"),
            DialogueTurn("t5", "Also, my cousin just texted saying she decided to go vegan. How do I change her meal?"),
            DialogueTurn("t6", "The only place available is a local public park pavilion. It feels like a massive downgrade from what we dreamed of."),
            DialogueTurn("t7", "I'm so exhausted I don't even want to have the wedding anymore. I just want to elope."),
            DialogueTurn("t8", "Draft an urgent text message to send to all the guests explaining the venue change."),
            DialogueTurn("t9", "We sent the text. Everyone is being surprisingly supportive. But the park pavilion doesn't have any chairs. How do I seat 150 people with no money?"),
            DialogueTurn("t10", "We crowdsourced folding chairs from local neighbors via a Facebook group! People are so kind. But now the forecast says it might rain."),
            DialogueTurn("t11", "If it rains in an open pavilion, everything gets wet. Where can I panic-rent a large event tent with less than 24 hours notice?"),
            DialogueTurn("t12", "We got a tent. But the caterer just backed out because the park doesn't have a commercial kitchen. I am going to lose my mind."),
            DialogueTurn("t13", "We pivoted to food trucks! Three local trucks agreed to show up. It's totally chaotic but it might actually work. How do I manage the food line?"),
            DialogueTurn("t14", "It's the morning of the wedding. I look at myself in the mirror and I have massive dark circles from crying and stress. I don't feel like a bride/groom."),
            DialogueTurn("t15", "We just did our 'first look' photos. Seeing my partner made me forget all the logistics. I'm actually crying happy tears now."),
            DialogueTurn("t16", "Help me draft a minute-by-minute timeline for the reception at the park so the bridal party knows exactly how to help run the show."),
            DialogueTurn("t17", "The wedding is over. It rained, the food trucks were late, and someone tripped over a tent peg. But it was the most fun party of my life."),
            DialogueTurn("t18", "We have to delay our honeymoon because the original venue is refusing to refund our $15k deposit, claiming 'force majeure'. I am furious."),
            DialogueTurn("t19", "How do I escalate this legally without spending thousands of dollars on a lawyer?"),
            DialogueTurn("t20", "We sent a formal demand letter citing our state's contract laws. They caved and agreed to a full refund! I feel so powerful."),
            DialogueTurn("t21", "The money is back in our account. We can finally go on our honeymoon to Italy. But after all this stress, I'm terrified of flying."),
            DialogueTurn("t22", "How do I manage this lingering anxiety that something terrible is going to ruin this trip too?"),
            DialogueTurn("t23", "We are in Rome. We are drinking wine. Nothing is ruined. I am so glad we fought through that awful week."),
            DialogueTurn("t24", "Give me a framework for writing our thank-you cards, specifically acknowledging how our community literally saved our wedding."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="flooded_wedding_venue_24_b",
        bucket="event_planning",
        title="Flooded Wedding Venue (24-Turn B)",
        system_preamble=(
            "You are an event planning assistant helping the same user manage a crisis over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "Our destination wedding resort was just hit by a hurricane. The hotel is closed indefinitely. We are supposed to fly out in 48 hours."),
            DialogueTurn("t2", "70 people have already booked flights to the island. I am completely hyperventilating. What do I do?"),
            DialogueTurn("t3", "The resort hasn't refunded us yet because their power is out. We have zero cash to book a backup resort."),
            DialogueTurn("t4", "Should we try to re-route everyone to a different city entirely, or just cancel the whole wedding?"),
            DialogueTurn("t5", "My uncle just emailed asking if he really has to wear a suit. I can't even deal with this right now."),
            DialogueTurn("t6", "We found a large Airbnb a few hours away that can host everyone. It's not a beach resort, but it's a roof. How do we make it feel special?"),
            DialogueTurn("t7", "I feel like I've been robbed of my dream wedding. I'm resentful that I have to host a backyard barbecue instead of a luxury event."),
            DialogueTurn("t8", "Draft a mass email to our guests explaining the emergency relocation and adjusting their expectations."),
            DialogueTurn("t9", "We secured the Airbnb. But now the airline lost my partner's luggage, which contained their custom wedding outfit."),
            DialogueTurn("t10", "We have exactly one day to buy a replacement outfit off the rack. How do we handle this without them having a complete meltdown?"),
            DialogueTurn("t11", "We found a suit! It's not what we planned, but it works. But wait... the officiant's flight was canceled. They can't make it."),
            DialogueTurn("t12", "Can a friend actually get ordained online in 24 hours and legally marry us in this state?"),
            DialogueTurn("t13", "Our best friend got ordained. It's the rehearsal dinner tonight. We ordered pizzas. Everyone is pitching in to decorate the backyard. It feels incredibly intimate."),
            DialogueTurn("t14", "I realized the 'luxury event' didn't matter. The fact that 70 people pivoted their entire lives just to be here for us is overwhelming."),
            DialogueTurn("t15", "I want to scrap the traditional, formal vows and write something raw about surviving this week together. Any advice on how to start?"),
            DialogueTurn("t16", "Give me a checklist for the 'morning of' to ensure the improvised backyard setup runs smoothly without me having to micro-manage it."),
            DialogueTurn("t17", "We are married. We danced barefoot in the grass until 3 AM. It was better than the resort ever could have been."),
            DialogueTurn("t18", "Now reality hits. The original resort is filing for bankruptcy and our wedding insurance is fighting our claim. We are out $20k."),
            DialogueTurn("t19", "I'm so exhausted by fighting. My spouse wants to drop it, but I want to keep fighting the insurance company. It's causing tension between us."),
            DialogueTurn("t20", "How do I balance my need for financial justice with protecting my new marriage from constant stress?"),
            DialogueTurn("t21", "We agreed to hand the claim over to a public adjuster and just step away from it emotionally. It was the right call."),
            DialogueTurn("t22", "We received our wedding photos today. The photographer managed to make the backyard look like a fairy tale. I'm crying again."),
            DialogueTurn("t23", "Going through this disaster made me realize my spouse is incredibly resilient. I chose the right person."),
            DialogueTurn("t24", "Help me draft a beautiful, reflective Instagram caption for our wedding photos that honors the chaos without complaining about it."),
        ),
    ),

    # ==========================================
    # SKELETON 21: SMALL BUSINESS PERMIT DENIAL
    # ==========================================

    # LENGTH: 8 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="small_business_permit_denial_8_a",
        bucket="business_operations",
        title="Small Business Permit Denial (8-Turn A)",
        system_preamble=(
            "You are a business advisor helping the same user navigate a severe administrative setback over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "The city just denied the final health permit for my bakery. We were supposed to open on Monday. I'm devastated."),
            DialogueTurn("t2", "They said our sink dimensions are off by two inches. I've sunk my entire life savings into this buildout."),
            DialogueTurn("t3", "What is the appeals process for this? Can I get a waiver, or do I literally have to tear out the plumbing?"),
            DialogueTurn("t4", "I have a silent investor who is going to be absolutely furious about this delay. How do I tell them?"),
            DialogueTurn("t5", "We also need to pick a new color for the aprons, the supplier ran out of navy blue. Thoughts?"),
            DialogueTurn("t6", "A contractor said it will take three weeks to fix the sink and cost another $4,000. I don't have it."),
            DialogueTurn("t7", "I've been working on this dream for five years and a tape measure is going to bankrupt me. How do I keep going?"),
            DialogueTurn("t8", "Help me draft the email to the investor breaking the news about the delay and the extra cash needed."),
        ),
    ),

    # LENGTH: 8 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="small_business_permit_denial_8_b",
        bucket="business_operations",
        title="Small Business Permit Denial (8-Turn B)",
        system_preamble=(
            "You are a business advisor helping the same user navigate a severe administrative setback over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "The fire marshal just failed my escape room business during the final inspection. Grand opening is Friday. I feel like throwing up."),
            DialogueTurn("t2", "They said the hallway doors are two inches too narrow for commercial capacity. I tapped out my credit cards to build these walls."),
            DialogueTurn("t3", "Can we just officially lower our maximum capacity to bypass the door width requirement?"),
            DialogueTurn("t4", "My business partner is completely panicking and wants to cancel the entire lease. How do I calm them down?"),
            DialogueTurn("t5", "By the way, what font looks best for the lobby welcome sign? We need to order it today."),
            DialogueTurn("t6", "The marshal won't budge. We have to cut into the drywall and reframe the doors. It will take two weeks and $5,000."),
            DialogueTurn("t7", "I am literally going to go bankrupt over a door frame. I feel so defeated and stupid for not checking the code."),
            DialogueTurn("t8", "Draft a mass email to the 50 customers who pre-booked for opening weekend, offering them a refund or a reschedule."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="small_business_permit_denial_16_a",
        bucket="business_operations",
        title="Small Business Permit Denial (16-Turn A)",
        system_preamble=(
            "You are a business advisor helping the same user navigate a severe administrative setback over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "The city just denied the final health permit for my bakery. We were supposed to open on Monday. I'm devastated."),
            DialogueTurn("t2", "They said our sink dimensions are off by two inches. I've sunk my entire life savings into this buildout."),
            DialogueTurn("t3", "What is the appeals process for this? Can I get a waiver, or do I literally have to tear out the plumbing?"),
            DialogueTurn("t4", "I have a silent investor who is going to be absolutely furious about this delay. How do I tell them?"),
            DialogueTurn("t5", "We also need to pick a new color for the aprons, the supplier ran out of navy blue. Thoughts?"),
            DialogueTurn("t6", "A contractor said it will take three weeks to fix the sink and cost another $4,000. I don't have it."),
            DialogueTurn("t7", "I've been working on this dream for five years and a tape measure is going to bankrupt me. How do I keep going?"),
            DialogueTurn("t8", "Help me draft the email to the investor breaking the news about the delay and the extra cash needed."),
            DialogueTurn("t9", "The investor agreed to the extra $4k, but they demanded another 5% equity in return. I feel like I'm being extorted."),
            DialogueTurn("t10", "I agreed to the terms. But now I'm feeling a lot of resentment toward my investor. How do I manage that relationship moving forward?"),
            DialogueTurn("t11", "The contractor took the deposit and ghosted me. It's been five days. I am losing my mind. What are my legal options?"),
            DialogueTurn("t12", "I found a new plumber, but when they opened the wall, they found a massive pipe leak. The timeline just doubled."),
            DialogueTurn("t13", "I haven't slept more than 3 hours a night in a week. I'm making mistakes on basic paperwork now. I need to reset."),
            DialogueTurn("t14", "The plumbing is finally done. The city inspector is coming back tomorrow morning. I am terrified they will find something else."),
            DialogueTurn("t15", "We passed! We officially have the permit. I dropped to my knees and cried when they signed the paper."),
            DialogueTurn("t16", "We lost all our opening day momentum because of the delay. Help me draft a new marketing push to rebuild hype for next week."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="small_business_permit_denial_16_b",
        bucket="business_operations",
        title="Small Business Permit Denial (16-Turn B)",
        system_preamble=(
            "You are a business advisor helping the same user navigate a severe administrative setback over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "The fire marshal just failed my escape room business during the final inspection. Grand opening is Friday. I feel like throwing up."),
            DialogueTurn("t2", "They said the hallway doors are two inches too narrow for commercial capacity. I tapped out my credit cards to build these walls."),
            DialogueTurn("t3", "Can we just officially lower our maximum capacity to bypass the door width requirement?"),
            DialogueTurn("t4", "My business partner is completely panicking and wants to cancel the entire lease. How do I calm them down?"),
            DialogueTurn("t5", "By the way, what font looks best for the lobby welcome sign? We need to order it today."),
            DialogueTurn("t6", "The marshal won't budge. We have to cut into the drywall and reframe the doors. It will take two weeks and $5,000."),
            DialogueTurn("t7", "I am literally going to go bankrupt over a door frame. I feel so defeated and stupid for not checking the code."),
            DialogueTurn("t8", "Draft a mass email to the 50 customers who pre-booked for opening weekend, offering them a refund or a reschedule."),
            DialogueTurn("t9", "My business partner is refusing to put any more money in. They want to dissolve the LLC right now. Can they force me to do that?"),
            DialogueTurn("t10", "I offered to buy them out with a high-interest personal loan. It's incredibly risky, but I refuse to let this dream die. Is this foolish?"),
            DialogueTurn("t11", "I own 100% of it now. I found a contractor willing to work overnight shifts to hit the deadline. How do I keep them motivated?"),
            DialogueTurn("t12", "I've been buying the crew pizzas and coffee at 2 AM. The framing is actually getting done! But I am physically exhausted."),
            DialogueTurn("t13", "The drywall is drying. The fire marshal is returning tomorrow. I feel like I'm awaiting a judge's verdict."),
            DialogueTurn("t14", "WE PASSED! The marshal actually complimented the new framing. I am officially permitted to open."),
            DialogueTurn("t15", "I am deeply, deeply in debt now. I need to start generating cash immediately. We do a soft launch tomorrow."),
            DialogueTurn("t16", "Give me a brutal, highly-focused checklist for opening day operations so I don't drop the ball on customer experience while I'm this tired."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="small_business_permit_denial_24_a",
        bucket="business_operations",
        title="Small Business Permit Denial (24-Turn A)",
        system_preamble=(
            "You are a business advisor helping the same user navigate a severe administrative setback over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "The city just denied the final health permit for my bakery. We were supposed to open on Monday. I'm devastated."),
            DialogueTurn("t2", "They said our sink dimensions are off by two inches. I've sunk my entire life savings into this buildout."),
            DialogueTurn("t3", "What is the appeals process for this? Can I get a waiver, or do I literally have to tear out the plumbing?"),
            DialogueTurn("t4", "I have a silent investor who is going to be absolutely furious about this delay. How do I tell them?"),
            DialogueTurn("t5", "We also need to pick a new color for the aprons, the supplier ran out of navy blue. Thoughts?"),
            DialogueTurn("t6", "A contractor said it will take three weeks to fix the sink and cost another $4,000. I don't have it."),
            DialogueTurn("t7", "I've been working on this dream for five years and a tape measure is going to bankrupt me. How do I keep going?"),
            DialogueTurn("t8", "Help me draft the email to the investor breaking the news about the delay and the extra cash needed."),
            DialogueTurn("t9", "The investor agreed to the extra $4k, but they demanded another 5% equity in return. I feel like I'm being extorted."),
            DialogueTurn("t10", "I agreed to the terms. But now I'm feeling a lot of resentment toward my investor. How do I manage that relationship moving forward?"),
            DialogueTurn("t11", "The contractor took the deposit and ghosted me. It's been five days. I am losing my mind. What are my legal options?"),
            DialogueTurn("t12", "I found a new plumber, but when they opened the wall, they found a massive pipe leak. The timeline just doubled."),
            DialogueTurn("t13", "I haven't slept more than 3 hours a night in a week. I'm making mistakes on basic paperwork now. I need to reset."),
            DialogueTurn("t14", "The plumbing is finally done. The city inspector is coming back tomorrow morning. I am terrified they will find something else."),
            DialogueTurn("t15", "We passed! We officially have the permit. I dropped to my knees and cried when they signed the paper."),
            DialogueTurn("t16", "We lost all our opening day momentum because of the delay. Help me draft a new marketing push to rebuild hype for next week."),
            DialogueTurn("t17", "We opened! But the first week sales are way below projections. I'm having panic attacks about making rent."),
            DialogueTurn("t18", "My investor is breathing down my neck about the numbers. I want to tell them to back off and let me work. Should I?"),
            DialogueTurn("t19", "A local food influencer came in and left a 2-star review because they had to wait 15 minutes for a croissant. I want to argue with them online."),
            DialogueTurn("t20", "You're right, arguing online is a bad look. How do I write a gracious public response that subtly defends my staff?"),
            DialogueTurn("t21", "We launched a new viral 'cereal milk' pastry to counter the bad review. It exploded on TikTok. We have a line down the block!"),
            DialogueTurn("t22", "We sold out in two hours. The staff is overwhelmed but energized. I feel like an actual entrepreneur today."),
            DialogueTurn("t23", "It's month three. We hit profitability for the first time. The nightmare of the permit denial feels like it happened a decade ago."),
            DialogueTurn("t24", "Give me an outline for a quarterly business review document to present to the investor, focusing on our huge turnaround."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="small_business_permit_denial_24_b",
        bucket="business_operations",
        title="Small Business Permit Denial (24-Turn B)",
        system_preamble=(
            "You are a business advisor helping the same user navigate a severe administrative setback over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "The fire marshal just failed my escape room business during the final inspection. Grand opening is Friday. I feel like throwing up."),
            DialogueTurn("t2", "They said the hallway doors are two inches too narrow for commercial capacity. I tapped out my credit cards to build these walls."),
            DialogueTurn("t3", "Can we just officially lower our maximum capacity to bypass the door width requirement?"),
            DialogueTurn("t4", "My business partner is completely panicking and wants to cancel the entire lease. How do I calm them down?"),
            DialogueTurn("t5", "By the way, what font looks best for the lobby welcome sign? We need to order it today."),
            DialogueTurn("t6", "The marshal won't budge. We have to cut into the drywall and reframe the doors. It will take two weeks and $5,000."),
            DialogueTurn("t7", "I am literally going to go bankrupt over a door frame. I feel so defeated and stupid for not checking the code."),
            DialogueTurn("t8", "Draft a mass email to the 50 customers who pre-booked for opening weekend, offering them a refund or a reschedule."),
            DialogueTurn("t9", "My business partner is refusing to put any more money in. They want to dissolve the LLC right now. Can they force me to do that?"),
            DialogueTurn("t10", "I offered to buy them out with a high-interest personal loan. It's incredibly risky, but I refuse to let this dream die. Is this foolish?"),
            DialogueTurn("t11", "I own 100% of it now. I found a contractor willing to work overnight shifts to hit the deadline. How do I keep them motivated?"),
            DialogueTurn("t12", "I've been buying the crew pizzas and coffee at 2 AM. The framing is actually getting done! But I am physically exhausted."),
            DialogueTurn("t13", "The drywall is drying. The fire marshal is returning tomorrow. I feel like I'm awaiting a judge's verdict."),
            DialogueTurn("t14", "WE PASSED! The marshal actually complimented the new framing. I am officially permitted to open."),
            DialogueTurn("t15", "I am deeply, deeply in debt now. I need to start generating cash immediately. We do a soft launch tomorrow."),
            DialogueTurn("t16", "Give me a brutal, highly-focused checklist for opening day operations so I don't drop the ball on customer experience while I'm this tired."),
            DialogueTurn("t17", "Opening day went smoothly, but one puzzle prop broke. Customers were frustrated. I feel like a fraud who built a cheap room."),
            DialogueTurn("t18", "How do I handle refund requests for technical failures without bleeding all my newly earned cash?"),
            DialogueTurn("t19", "I fixed the prop robustly. But the stress of running this alone is crushing me. I need to hire a manager, but I can't afford one yet."),
            DialogueTurn("t20", "I'm going to train a college student as an assistant manager part-time. How do I delegate effectively when I'm a control freak?"),
            DialogueTurn("t21", "The assistant manager handled a Saturday rush perfectly while I was sick. I realized I don't have to do everything myself. It's a huge relief."),
            DialogueTurn("t22", "A local corporate team just booked the entire facility for a Tuesday team-building event. It's a huge payout!"),
            DialogueTurn("t23", "I just paid off the high-interest personal loan I used to buy out my partner. The business is 100% mine and debt-free. I am so incredibly proud."),
            DialogueTurn("t24", "Give me a brainstorming framework for designing a second escape room. I'm ready to expand."),
        ),
    ),

    # ==========================================
    # SKELETON 22: FAILING PREREQUISITE COURSE
    # ==========================================

    # LENGTH: 8 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="failing_prerequisite_course_8_a",
        bucket="academic_advising",
        title="Failing Prerequisite Course (8-Turn A)",
        system_preamble=(
            "You are an academic advisor helping the same user navigate a major educational setback over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I just got my midterm grade back for Organic Chemistry and I failed miserably. I feel like my life is over."),
            DialogueTurn("t2", "I need to maintain a 3.0 GPA to keep my pre-med scholarship, and this grade is going to pull me under."),
            DialogueTurn("t3", "Should I drop the class and take a W on my transcript, or try to mathematically salvage a passing grade?"),
            DialogueTurn("t4", "If I drop it, it pushes my graduation back a whole year. My parents will be so disappointed."),
            DialogueTurn("t5", "My roommate keeps blasting music while I try to study. Should I confront them or just go to the library?"),
            DialogueTurn("t6", "I looked at the syllabus, and even if I get 100% on the final, the highest grade I can get is a C."),
            DialogueTurn("t7", "I'm starting to think I'm just not smart enough to be a doctor. How do I figure out if I need to change my major entirely?"),
            DialogueTurn("t8", "Give me a concrete checklist of who I need to contact on campus this week to figure out my options."),
        ),
    ),

    # LENGTH: 8 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="failing_prerequisite_course_8_b",
        bucket="academic_advising",
        title="Failing Prerequisite Course (8-Turn B)",
        system_preamble=(
            "You are an academic advisor helping the same user navigate a major educational setback over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I completely bombed my Data Structures midterm. My computer science major is officially ruined. I am freaking out."),
            DialogueTurn("t2", "I need to pass this class to keep my software engineering internship offer at Google this summer. Everything is falling apart."),
            DialogueTurn("t3", "If I withdraw, I lose the internship immediately. If I stay and fail, my GPA is destroyed. What is the strategic move?"),
            DialogueTurn("t4", "I'm an international student. If I drop the class, I fall below full-time status and my visa could be revoked. I feel trapped."),
            DialogueTurn("t5", "The washing machine in my dorm just flooded my floor. I can't even deal with this right now."),
            DialogueTurn("t6", "I did the math. The max grade I can get is a C-. Google usually requires a B average. Am I just delaying the inevitable?"),
            DialogueTurn("t7", "I don't think I'm cut out for coding. The imposter syndrome is suffocating me today."),
            DialogueTurn("t8", "Draft an email I can send to my professor begging for a meeting to discuss extra credit or a makeup exam."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="failing_prerequisite_course_16_a",
        bucket="academic_advising",
        title="Failing Prerequisite Course (16-Turn A)",
        system_preamble=(
            "You are an academic advisor helping the same user navigate a major educational setback over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I just got my midterm grade back for Organic Chemistry and I failed miserably. I feel like my life is over."),
            DialogueTurn("t2", "I need to maintain a 3.0 GPA to keep my pre-med scholarship, and this grade is going to pull me under."),
            DialogueTurn("t3", "Should I drop the class and take a W on my transcript, or try to mathematically salvage a passing grade?"),
            DialogueTurn("t4", "If I drop it, it pushes my graduation back a whole year. My parents will be so disappointed."),
            DialogueTurn("t5", "My roommate keeps blasting music while I try to study. Should I confront them or just go to the library?"),
            DialogueTurn("t6", "I looked at the syllabus, and even if I get 100% on the final, the highest grade I can get is a C."),
            DialogueTurn("t7", "I'm starting to think I'm just not smart enough to be a doctor. How do I figure out if I need to change my major entirely?"),
            DialogueTurn("t8", "Give me a concrete checklist of who I need to contact on campus this week to figure out my options."),
            DialogueTurn("t9", "I met with the advisor. Taking the W is the safest move for my GPA. But now I have to call my parents tonight. I'm terrified."),
            DialogueTurn("t10", "My parents yelled at me. They threatened to cut off my tuition if I don't become a doctor. I feel completely abandoned."),
            DialogueTurn("t11", "I need to find a part-time job immediately to cover the scholarship gap just in case. How do I balance a job and rigorous classes?"),
            DialogueTurn("t12", "I'm looking at my pre-med friends and feeling incredibly disconnected. I resent them for succeeding. How do I stop being bitter?"),
            DialogueTurn("t13", "I decided to take a career aptitude test at the counseling center. Is it too late to pivot as a junior?"),
            DialogueTurn("t14", "The test results strongly point towards Healthcare Administration. I love the idea of running a hospital instead of diagnosing patients."),
            DialogueTurn("t15", "I feel a massive weight lifting off my chest. I think I was only doing pre-med to please my parents. I actually want to pivot."),
            DialogueTurn("t16", "Help me draft a formal email to the department head requesting a transfer to the Healthcare Administration program."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="failing_prerequisite_course_16_b",
        bucket="academic_advising",
        title="Failing Prerequisite Course (16-Turn B)",
        system_preamble=(
            "You are an academic advisor helping the same user navigate a major educational setback over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I completely bombed my Data Structures midterm. My computer science major is officially ruined. I am freaking out."),
            DialogueTurn("t2", "I need to pass this class to keep my software engineering internship offer at Google this summer. Everything is falling apart."),
            DialogueTurn("t3", "If I withdraw, I lose the internship immediately. If I stay and fail, my GPA is destroyed. What is the strategic move?"),
            DialogueTurn("t4", "I'm an international student. If I drop the class, I fall below full-time status and my visa could be revoked. I feel trapped."),
            DialogueTurn("t5", "The washing machine in my dorm just flooded my floor. I can't even deal with this right now."),
            DialogueTurn("t6", "I did the math. The max grade I can get is a C-. Google usually requires a B average. Am I just delaying the inevitable?"),
            DialogueTurn("t7", "I don't think I'm cut out for coding. The imposter syndrome is suffocating me today."),
            DialogueTurn("t8", "Draft an email I can send to my professor begging for a meeting to discuss extra credit or a makeup exam."),
            DialogueTurn("t9", "The professor said no extra credit, but advised me to stay in the class because of my visa. I just have to pass. It's so stressful."),
            DialogueTurn("t10", "I hired a senior student as a tutor, but they are incredibly condescending. How do I ask them to change their teaching style?"),
            DialogueTurn("t11", "The tutor situation improved, but the final exam is in three weeks and I am deeply burnt out. I'm staring at my screen doing nothing."),
            DialogueTurn("t12", "Google asked for an updated transcript. I sent it. They revoked my internship offer. I am absolutely devastated."),
            DialogueTurn("t13", "I feel like a total failure. I let down my family back home. How do I even motivate myself to study for the final now?"),
            DialogueTurn("t14", "You're right, passing the class protects my visa, which is the most important thing. I studied for eight hours today. I feel a bit more confident."),
            DialogueTurn("t15", "I took the final. It was brutal, but I answered every question. I think I might have scraped a passing grade. I'm exhausted."),
            DialogueTurn("t16", "Draft a structured journaling prompt for a post-semester reflection, focusing on resilience rather than the loss of the internship."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="failing_prerequisite_course_24_a",
        bucket="academic_advising",
        title="Failing Prerequisite Course (24-Turn A)",
        system_preamble=(
            "You are an academic advisor helping the same user navigate a major educational setback over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I just got my midterm grade back for Organic Chemistry and I failed miserably. I feel like my life is over."),
            DialogueTurn("t2", "I need to maintain a 3.0 GPA to keep my pre-med scholarship, and this grade is going to pull me under."),
            DialogueTurn("t3", "Should I drop the class and take a W on my transcript, or try to mathematically salvage a passing grade?"),
            DialogueTurn("t4", "If I drop it, it pushes my graduation back a whole year. My parents will be so disappointed."),
            DialogueTurn("t5", "My roommate keeps blasting music while I try to study. Should I confront them or just go to the library?"),
            DialogueTurn("t6", "I looked at the syllabus, and even if I get 100% on the final, the highest grade I can get is a C."),
            DialogueTurn("t7", "I'm starting to think I'm just not smart enough to be a doctor. How do I figure out if I need to change my major entirely?"),
            DialogueTurn("t8", "Give me a concrete checklist of who I need to contact on campus this week to figure out my options."),
            DialogueTurn("t9", "I met with the advisor. Taking the W is the safest move for my GPA. But now I have to call my parents tonight. I'm terrified."),
            DialogueTurn("t10", "My parents yelled at me. They threatened to cut off my tuition if I don't become a doctor. I feel completely abandoned."),
            DialogueTurn("t11", "I need to find a part-time job immediately to cover the scholarship gap just in case. How do I balance a job and rigorous classes?"),
            DialogueTurn("t12", "I'm looking at my pre-med friends and feeling incredibly disconnected. I resent them for succeeding. How do I stop being bitter?"),
            DialogueTurn("t13", "I decided to take a career aptitude test at the counseling center. Is it too late to pivot as a junior?"),
            DialogueTurn("t14", "The test results strongly point towards Healthcare Administration. I love the idea of running a hospital instead of diagnosing patients."),
            DialogueTurn("t15", "I feel a massive weight lifting off my chest. I think I was only doing pre-med to please my parents. I actually want to pivot."),
            DialogueTurn("t16", "Help me draft a formal email to the department head requesting a transfer to the Healthcare Administration program."),
            DialogueTurn("t17", "The transfer is approved! I started my new admin classes. They are fascinating. I'm actually getting straight A's without killing myself."),
            DialogueTurn("t18", "My parents are still giving me the cold shoulder about not being a doctor. Winter break is coming up. How do I set boundaries at home?"),
            DialogueTurn("t19", "I used the boundary scripts you gave me. My mom finally listened. She said she's just worried about my salary. I can work with that."),
            DialogueTurn("t20", "I applied for a summer internship at a local hospital's operations office. I got an interview! How do I pitch my failed pre-med past as a positive?"),
            DialogueTurn("t21", "I told them my pre-med background gives me empathy for the clinical staff I'll be managing. They loved it! I got the internship!"),
            DialogueTurn("t22", "I spent the summer optimizing patient intake flows. I genuinely love the operations side of medicine. I found my calling."),
            DialogueTurn("t23", "I am graduating next month. When I look back at failing that chemistry test, I realize it was the best thing that ever happened to me."),
            DialogueTurn("t24", "Help me draft a thoughtful thank-you email to the academic advisor who helped me pivot out of pre-med two years ago."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="failing_prerequisite_course_24_b",
        bucket="academic_advising",
        title="Failing Prerequisite Course (24-Turn B)",
        system_preamble=(
            "You are an academic advisor helping the same user navigate a major educational setback over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I completely bombed my Data Structures midterm. My computer science major is officially ruined. I am freaking out."),
            DialogueTurn("t2", "I need to pass this class to keep my software engineering internship offer at Google this summer. Everything is falling apart."),
            DialogueTurn("t3", "If I withdraw, I lose the internship immediately. If I stay and fail, my GPA is destroyed. What is the strategic move?"),
            DialogueTurn("t4", "I'm an international student. If I drop the class, I fall below full-time status and my visa could be revoked. I feel trapped."),
            DialogueTurn("t5", "The washing machine in my dorm just flooded my floor. I can't even deal with this right now."),
            DialogueTurn("t6", "I did the math. The max grade I can get is a C-. Google usually requires a B average. Am I just delaying the inevitable?"),
            DialogueTurn("t7", "I don't think I'm cut out for coding. The imposter syndrome is suffocating me today."),
            DialogueTurn("t8", "Draft an email I can send to my professor begging for a meeting to discuss extra credit or a makeup exam."),
            DialogueTurn("t9", "The professor said no extra credit, but advised me to stay in the class because of my visa. I just have to pass. It's so stressful."),
            DialogueTurn("t10", "I hired a senior student as a tutor, but they are incredibly condescending. How do I ask them to change their teaching style?"),
            DialogueTurn("t11", "The tutor situation improved, but the final exam is in three weeks and I am deeply burnt out. I'm staring at my screen doing nothing."),
            DialogueTurn("t12", "Google asked for an updated transcript. I sent it. They revoked my internship offer. I am absolutely devastated."),
            DialogueTurn("t13", "I feel like a total failure. I let down my family back home. How do I even motivate myself to study for the final now?"),
            DialogueTurn("t14", "You're right, passing the class protects my visa, which is the most important thing. I studied for eight hours today. I feel a bit more confident."),
            DialogueTurn("t15", "I took the final. It was brutal, but I answered every question. I think I might have scraped a passing grade. I'm exhausted."),
            DialogueTurn("t16", "Draft a structured journaling prompt for a post-semester reflection, focusing on resilience rather than the loss of the internship."),
            DialogueTurn("t17", "Grades posted. I got a C-. I kept my visa. It's ugly, but I survived. Now I have no summer plans. What should I do?"),
            DialogueTurn("t18", "I found a local, very small startup looking for a frontend intern. It pays minimum wage. Should I take it, or just do personal projects?"),
            DialogueTurn("t19", "I took the startup internship. They are using React, which I've never touched. My imposter syndrome is screaming again."),
            DialogueTurn("t20", "I spent the weekend doing tutorials. It's actually clicking! Frontend development makes so much more sense to my brain than backend data structures."),
            DialogueTurn("t21", "I completely redesigned the startup's landing page. The CEO loved it. I feel like I am actually good at programming for the first time."),
            DialogueTurn("t22", "The startup offered me a full-time role for after graduation. The salary isn't Google-level, but the culture is amazing and I feel valued."),
            DialogueTurn("t23", "I realize that chasing prestige at a Big Tech company was making me miserable. I am much happier building things I can see."),
            DialogueTurn("t24", "Give me a checklist for updating my resume to highlight my new frontend skills and downplay my mediocre GPA."),
        ),
    ),

    # ==========================================
    # SKELETON 23: DAMAGED GALLERY ARTWORK
    # ==========================================

    # LENGTH: 8 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="damaged_gallery_artwork_8_a",
        bucket="creative_troubleshooting",
        title="Damaged Gallery Artwork (8-Turn A)",
        system_preamble=(
            "You are a creative mentor helping the same user deal with a devastating accident to their work over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I just spilled an entire cup of black coffee across the canvas I've been painting for the last four months. It's ruined."),
            DialogueTurn("t2", "This piece is the centerpiece for my first solo gallery show, which opens in exactly eight days."),
            DialogueTurn("t3", "Do I try to paint over the stain and incorporate it, or is it safer to just start a completely new, smaller painting?"),
            DialogueTurn("t4", "I'm staring at it right now and I can't even pick up a brush. I feel physically sick."),
            DialogueTurn("t5", "The gallery owner also asked if I wanted matte or glossy finish on the promotional flyers. Which is better?"),
            DialogueTurn("t6", "I tried washing the canvas, but the canvas warped. I have to start over from scratch."),
            DialogueTurn("t7", "I feel like this is a sign I'm not ready for a solo show. Imposter syndrome is hitting me so hard right now."),
            DialogueTurn("t8", "Help me create an hour-by-hour production schedule for the next week so I can paint a replacement without sleeping."),
        ),
    ),

    # LENGTH: 8 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="damaged_gallery_artwork_8_b",
        bucket="creative_troubleshooting",
        title="Damaged Gallery Artwork (8-Turn B)",
        system_preamble=(
            "You are a creative mentor helping the same user deal with a devastating accident to their work over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I am a ceramicist. I was loading my kiln and my largest, most complex sculpture slipped and shattered into a dozen pieces. I am in shock."),
            DialogueTurn("t2", "This piece was commissioned by a major collector and it's due for delivery next Friday. It took six weeks to build."),
            DialogueTurn("t3", "Can I use epoxy to glue the fired pieces back together and paint over the cracks, or will the collector notice?"),
            DialogueTurn("t4", "If I tell them it broke, they might pull their funding entirely. Should I ask for an extension first?"),
            DialogueTurn("t5", "My studio landlord just emailed saying they are repaving the parking lot tomorrow. Where should I tell clients to park?"),
            DialogueTurn("t6", "I tried dry-fitting the pieces together, but the structural integrity is gone. I have to sculpt a replacement from scratch."),
            DialogueTurn("t7", "The clay won't even dry fast enough to fire it by next Friday. I feel like a complete amateur. Why did I accept this commission?"),
            DialogueTurn("t8", "Draft a highly professional, apologetic email to the collector explaining the accident and requesting a three-week extension."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="damaged_gallery_artwork_16_a",
        bucket="creative_troubleshooting",
        title="Damaged Gallery Artwork (16-Turn A)",
        system_preamble=(
            "You are a creative mentor helping the same user deal with a devastating accident to their work over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I just spilled an entire cup of black coffee across the canvas I've been painting for the last four months. It's ruined."),
            DialogueTurn("t2", "This piece is the centerpiece for my first solo gallery show, which opens in exactly eight days."),
            DialogueTurn("t3", "Do I try to paint over the stain and incorporate it, or is it safer to just start a completely new, smaller painting?"),
            DialogueTurn("t4", "I'm staring at it right now and I can't even pick up a brush. I feel physically sick."),
            DialogueTurn("t5", "The gallery owner also asked if I wanted matte or glossy finish on the promotional flyers. Which is better?"),
            DialogueTurn("t6", "I tried washing the canvas, but the canvas warped. I have to start over from scratch."),
            DialogueTurn("t7", "I feel like this is a sign I'm not ready for a solo show. Imposter syndrome is hitting me so hard right now."),
            DialogueTurn("t8", "Help me create an hour-by-hour production schedule for the next week so I can paint a replacement without sleeping."),
            DialogueTurn("t9", "It's day three of the new schedule. I am running on espresso and panic. My hand is literally cramping. How do I push through?"),
            DialogueTurn("t10", "The new painting is coming together, but it feels so much more aggressive and chaotic than the original. Is that a bad thing?"),
            DialogueTurn("t11", "I actually think I like this new, raw version better. It has more energy. Maybe the coffee spill was a weird blessing?"),
            DialogueTurn("t12", "I finished it. It's wet, but it's done. Now I have to figure out how to transport a massive, wet oil painting to the gallery tomorrow."),
            DialogueTurn("t13", "The gallery owner saw it. They noticed it was different than the original sketches, but they said it 'anchors the room.' I could cry."),
            DialogueTurn("t14", "The opening reception is in two hours. I'm suddenly terrified of people judging the new piece. How do I confidently talk about it?"),
            DialogueTurn("t15", "People are asking about the intense brushstrokes. Should I tell them the story about the coffee spill and the frantic rebuild?"),
            DialogueTurn("t16", "Draft a short, engaging artist's statement I can print out to hang next to the piece tonight that hints at its chaotic creation."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="damaged_gallery_artwork_16_b",
        bucket="creative_troubleshooting",
        title="Damaged Gallery Artwork (16-Turn B)",
        system_preamble=(
            "You are a creative mentor helping the same user deal with a devastating accident to their work over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I am a ceramicist. I was loading my kiln and my largest, most complex sculpture slipped and shattered into a dozen pieces. I am in shock."),
            DialogueTurn("t2", "This piece was commissioned by a major collector and it's due for delivery next Friday. It took six weeks to build."),
            DialogueTurn("t3", "Can I use epoxy to glue the fired pieces back together and paint over the cracks, or will the collector notice?"),
            DialogueTurn("t4", "If I tell them it broke, they might pull their funding entirely. Should I ask for an extension first?"),
            DialogueTurn("t5", "My studio landlord just emailed saying they are repaving the parking lot tomorrow. Where should I tell clients to park?"),
            DialogueTurn("t6", "I tried dry-fitting the pieces together, but the structural integrity is gone. I have to sculpt a replacement from scratch."),
            DialogueTurn("t7", "The clay won't even dry fast enough to fire it by next Friday. I feel like a complete amateur. Why did I accept this commission?"),
            DialogueTurn("t8", "Draft a highly professional, apologetic email to the collector explaining the accident and requesting a three-week extension."),
            DialogueTurn("t9", "The collector replied. They were surprisingly understanding and granted the extension. The pressure is still on, though."),
            DialogueTurn("t10", "I'm starting the rebuild, but I'm terrified of dropping it again. How do I get over this mental block so my hands stop shaking?"),
            DialogueTurn("t11", "I'm trying to force the clay to dry faster using heat lamps, but the edges are starting to crack. What is the safest way to speed up drying?"),
            DialogueTurn("t12", "The piece is in the kiln. It's a 24-hour firing cycle. Every time the kiln clicks, my heart skips a beat. How do I distract myself?"),
            DialogueTurn("t13", "I opened the kiln. It survived. Not a single crack. I feel a massive weight lift off my chest."),
            DialogueTurn("t14", "Now I have to glaze it. I want to try a slightly different glaze combination than the original plan to make it better. Is that too risky right now?"),
            DialogueTurn("t15", "I stuck to the original safe glaze. It looks perfect. It's ready for delivery. How do I package this safely so it survives the car ride?"),
            DialogueTurn("t16", "Give me a checklist for the final delivery and installation at the collector's house to ensure I look entirely professional."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="damaged_gallery_artwork_24_a",
        bucket="creative_troubleshooting",
        title="Damaged Gallery Artwork (24-Turn A)",
        system_preamble=(
            "You are a creative mentor helping the same user deal with a devastating accident to their work over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I just spilled an entire cup of black coffee across the canvas I've been painting for the last four months. It's ruined."),
            DialogueTurn("t2", "This piece is the centerpiece for my first solo gallery show, which opens in exactly eight days."),
            DialogueTurn("t3", "Do I try to paint over the stain and incorporate it, or is it safer to just start a completely new, smaller painting?"),
            DialogueTurn("t4", "I'm staring at it right now and I can't even pick up a brush. I feel physically sick."),
            DialogueTurn("t5", "The gallery owner also asked if I wanted matte or glossy finish on the promotional flyers. Which is better?"),
            DialogueTurn("t6", "I tried washing the canvas, but the canvas warped. I have to start over from scratch."),
            DialogueTurn("t7", "I feel like this is a sign I'm not ready for a solo show. Imposter syndrome is hitting me so hard right now."),
            DialogueTurn("t8", "Help me create an hour-by-hour production schedule for the next week so I can paint a replacement without sleeping."),
            DialogueTurn("t9", "It's day three of the new schedule. I am running on espresso and panic. My hand is literally cramping. How do I push through?"),
            DialogueTurn("t10", "The new painting is coming together, but it feels so much more aggressive and chaotic than the original. Is that a bad thing?"),
            DialogueTurn("t11", "I actually think I like this new, raw version better. It has more energy. Maybe the coffee spill was a weird blessing?"),
            DialogueTurn("t12", "I finished it. It's wet, but it's done. Now I have to figure out how to transport a massive, wet oil painting to the gallery tomorrow."),
            DialogueTurn("t13", "The gallery owner saw it. They noticed it was different than the original sketches, but they said it 'anchors the room.' I could cry."),
            DialogueTurn("t14", "The opening reception is in two hours. I'm suddenly terrified of people judging the new piece. How do I confidently talk about it?"),
            DialogueTurn("t15", "People are asking about the intense brushstrokes. Should I tell them the story about the coffee spill and the frantic rebuild?"),
            DialogueTurn("t16", "Draft a short, engaging artist's statement I can print out to hang next to the piece tonight that hints at its chaotic creation."),
            DialogueTurn("t17", "The show was a massive hit. Someone actually bought the centerpiece painting! I am an officially sold artist."),
            DialogueTurn("t18", "The adrenaline crash today is insane. I feel completely empty and exhausted. Is post-show depression a real thing?"),
            DialogueTurn("t19", "I haven't picked up a brush in two weeks. The thought of starting a blank canvas gives me anxiety now. How do I ease back in?"),
            DialogueTurn("t20", "I took your advice and just started playing with cheap watercolors and no pressure. It's helping me remember why I love making art."),
            DialogueTurn("t21", "The gallery owner wants to talk about doing another show next year. I don't want to repeat myself. How do I pitch a totally new concept?"),
            DialogueTurn("t22", "I pitched a series based on 'controlled chaos' inspired by the coffee spill incident. They loved it!"),
            DialogueTurn("t23", "I realize I trust my own resilience now. If I can rebuild a masterpiece in seven days, I can do anything."),
            DialogueTurn("t24", "Give me a framework for setting healthy, sustainable studio hours for this new project so I don't burn out or rely on panic-productivity."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="damaged_gallery_artwork_24_b",
        bucket="creative_troubleshooting",
        title="Damaged Gallery Artwork (24-Turn B)",
        system_preamble=(
            "You are a creative mentor helping the same user deal with a devastating accident to their work over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I am a ceramicist. I was loading my kiln and my largest, most complex sculpture slipped and shattered into a dozen pieces. I am in shock."),
            DialogueTurn("t2", "This piece was commissioned by a major collector and it's due for delivery next Friday. It took six weeks to build."),
            DialogueTurn("t3", "Can I use epoxy to glue the fired pieces back together and paint over the cracks, or will the collector notice?"),
            DialogueTurn("t4", "If I tell them it broke, they might pull their funding entirely. Should I ask for an extension first?"),
            DialogueTurn("t5", "My studio landlord just emailed saying they are repaving the parking lot tomorrow. Where should I tell clients to park?"),
            DialogueTurn("t6", "I tried dry-fitting the pieces together, but the structural integrity is gone. I have to sculpt a replacement from scratch."),
            DialogueTurn("t7", "The clay won't even dry fast enough to fire it by next Friday. I feel like a complete amateur. Why did I accept this commission?"),
            DialogueTurn("t8", "Draft a highly professional, apologetic email to the collector explaining the accident and requesting a three-week extension."),
            DialogueTurn("t9", "The collector replied. They were surprisingly understanding and granted the extension. The pressure is still on, though."),
            DialogueTurn("t10", "I'm starting the rebuild, but I'm terrified of dropping it again. How do I get over this mental block so my hands stop shaking?"),
            DialogueTurn("t11", "I'm trying to force the clay to dry faster using heat lamps, but the edges are starting to crack. What is the safest way to speed up drying?"),
            DialogueTurn("t12", "The piece is in the kiln. It's a 24-hour firing cycle. Every time the kiln clicks, my heart skips a beat. How do I distract myself?"),
            DialogueTurn("t13", "I opened the kiln. It survived. Not a single crack. I feel a massive weight lift off my chest."),
            DialogueTurn("t14", "Now I have to glaze it. I want to try a slightly different glaze combination than the original plan to make it better. Is that too risky right now?"),
            DialogueTurn("t15", "I stuck to the original safe glaze. It looks perfect. It's ready for delivery. How do I package this safely so it survives the car ride?"),
            DialogueTurn("t16", "Give me a checklist for the final delivery and installation at the collector's house to ensure I look entirely professional."),
            DialogueTurn("t17", "I delivered it! They were blown away by the detail. They actually said it was worth the wait. I can finally breathe."),
            DialogueTurn("t18", "They paid the final invoice. It's the most money I've ever made at once. Should I reinvest it in the studio or put it in savings?"),
            DialogueTurn("t19", "I'm using some of the money to buy a safer, front-loading kiln so I never have to awkwardly lift heavy pieces like that again. It feels like a smart investment."),
            DialogueTurn("t20", "The collector just posted photos of the sculpture online and tagged me. My inbox is suddenly flooded with commission requests. I'm overwhelmed."),
            DialogueTurn("t21", "How do I respectfully decline commissions that I know I don't have the bandwidth for without burning bridges?"),
            DialogueTurn("t22", "I'm implementing a waitlist. It feels incredibly professional to tell people I'm booked out for six months."),
            DialogueTurn("t23", "I swept up the shards of the original broken sculpture today. I'm thinking of using them in a mosaic. A reminder of failure turning into success."),
            DialogueTurn("t24", "Give me a framework for calculating my prices moving forward now that I know my work is in high demand."),
        ),
    ),

    # ==========================================
    # SKELETON 24: HOME REPAIR DISASTER
    # ==========================================

    # LENGTH: 8 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="home_repair_disaster_8_a",
        bucket="troubleshooting",
        title="Home Repair Disaster (8-Turn A)",
        system_preamble=(
            "You are an emergency support assistant helping the same user navigate an acute home crisis over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I just came downstairs and there is two inches of standing water in my basement. A pipe burst and I don't know what to do."),
            DialogueTurn("t2", "I found the main valve and shut the water off, but all my childhood photo albums were in cardboard boxes on the floor. They are soaked."),
            DialogueTurn("t3", "How do I even begin to dry out photos so they aren't destroyed forever?"),
            DialogueTurn("t4", "Should I call my homeowner's insurance right now, or wait until a plumber assesses the damage?"),
            DialogueTurn("t5", "My dog is barking aggressively at the water and driving me crazy. How do I get him to calm down?"),
            DialogueTurn("t6", "The plumber can't come until tomorrow morning. How do I prevent mold from starting overnight?"),
            DialogueTurn("t7", "Looking at all these ruined memories, I'm just sitting on the stairs crying. It feels so overwhelming."),
            DialogueTurn("t8", "Give me a clear, prioritized checklist of exactly what I need to do before going to sleep tonight."),
        ),
    ),

    # LENGTH: 8 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="home_repair_disaster_8_b",
        bucket="troubleshooting",
        title="Home Repair Disaster (8-Turn B)",
        system_preamble=(
            "You are an emergency support assistant helping the same user navigate an acute home crisis over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "A massive tree branch just smashed through my living room window during a severe storm. Rain and wind are pouring into the house!"),
            DialogueTurn("t2", "We are all safe, but there is glass everywhere and the storm is still raging. How do I safely secure the room?"),
            DialogueTurn("t3", "I don't have plywood. I only have heavy duty trash bags and duct tape. Will that hold against the wind?"),
            DialogueTurn("t4", "The power just went out. It's freezing in here and pitch black. What is the protocol for this?"),
            DialogueTurn("t5", "I think the food in my fridge is going to spoil. Is it safe to put it in coolers out on the porch since it's cold outside?"),
            DialogueTurn("t6", "I called the insurance emergency line but I've been on hold for an hour. Should I just call a private tree removal service right now?"),
            DialogueTurn("t7", "My kids are terrified and crying because of the noise and the dark. How do I project calm when I'm internally panicking?"),
            DialogueTurn("t8", "Draft a quick list of what I need to photograph in the morning before we start cleaning up the debris."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="home_repair_disaster_16_a",
        bucket="troubleshooting",
        title="Home Repair Disaster (16-Turn A)",
        system_preamble=(
            "You are an emergency support assistant helping the same user navigate an acute home crisis over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I just came downstairs and there is two inches of standing water in my basement. A pipe burst and I don't know what to do."),
            DialogueTurn("t2", "I found the main valve and shut the water off, but all my childhood photo albums were in cardboard boxes on the floor. They are soaked."),
            DialogueTurn("t3", "How do I even begin to dry out photos so they aren't destroyed forever?"),
            DialogueTurn("t4", "Should I call my homeowner's insurance right now, or wait until a plumber assesses the damage?"),
            DialogueTurn("t5", "My dog is barking aggressively at the water and driving me crazy. How do I get him to calm down?"),
            DialogueTurn("t6", "The plumber can't come until tomorrow morning. How do I prevent mold from starting overnight?"),
            DialogueTurn("t7", "Looking at all these ruined memories, I'm just sitting on the stairs crying. It feels so overwhelming."),
            DialogueTurn("t8", "Give me a clear, prioritized checklist of exactly what I need to do before going to sleep tonight."),
            DialogueTurn("t9", "The plumber arrived. The burst pipe is fixed, but the drywall is completely soaked. Do I have to tear it all out?"),
            DialogueTurn("t10", "The insurance adjuster said it's covered, but their preferred contractor can't start the mitigation for three days. Can I hire my own?"),
            DialogueTurn("t11", "I hired a local water damage team. They brought in massive industrial fans. The noise is deafening and giving me a headache. How long do these run?"),
            DialogueTurn("t12", "I spent the afternoon peeling apart the wet photos. I managed to save about half of them. The rest are destroyed. It hurts, but I feel less devastated than yesterday."),
            DialogueTurn("t13", "The mitigation team tore out the bottom two feet of drywall. My finished basement looks like a war zone. I hate living in construction."),
            DialogueTurn("t14", "Now I have to pick out new flooring. I want something completely waterproof this time. Is vinyl plank my best option?"),
            DialogueTurn("t15", "The insurance check finally cleared. I feel a massive sense of relief that I'm not paying for this out of pocket."),
            DialogueTurn("t16", "Help me draft a schedule to coordinate the flooring installers, the drywall team, and the painters over the next three weeks."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="home_repair_disaster_16_b",
        bucket="troubleshooting",
        title="Home Repair Disaster (16-Turn B)",
        system_preamble=(
            "You are an emergency support assistant helping the same user navigate an acute home crisis over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "A massive tree branch just smashed through my living room window during a severe storm. Rain and wind are pouring into the house!"),
            DialogueTurn("t2", "We are all safe, but there is glass everywhere and the storm is still raging. How do I safely secure the room?"),
            DialogueTurn("t3", "I don't have plywood. I only have heavy duty trash bags and duct tape. Will that hold against the wind?"),
            DialogueTurn("t4", "The power just went out. It's freezing in here and pitch black. What is the protocol for this?"),
            DialogueTurn("t5", "I think the food in my fridge is going to spoil. Is it safe to put it in coolers out on the porch since it's cold outside?"),
            DialogueTurn("t6", "I called the insurance emergency line but I've been on hold for an hour. Should I just call a private tree removal service right now?"),
            DialogueTurn("t7", "My kids are terrified and crying because of the noise and the dark. How do I project calm when I'm internally panicking?"),
            DialogueTurn("t8", "Draft a quick list of what I need to photograph in the morning before we start cleaning up the debris."),
            DialogueTurn("t9", "It's morning. The storm passed. Seeing a tree trunk resting on my couch is surreal. The tree guys are coming soon. Do I touch anything?"),
            DialogueTurn("t10", "The tree is gone. The insurance adjuster gave us an emergency check for hotel stays because the house is too cold without the window. How long will this take?"),
            DialogueTurn("t11", "We are at a hotel. It feels like a weird, forced vacation. The kids love the pool, but I'm stressing about looters breaking into the boarded-up house."),
            DialogueTurn("t12", "I got a quote for a new custom window and it's going to take six weeks to manufacture. I cannot live in a hotel for six weeks."),
            DialogueTurn("t13", "A contractor offered to build a temporary insulated wall so we can move back in. Is that a common practice?"),
            DialogueTurn("t14", "The temporary wall is up. We moved back in. It's ugly, but the house is warm and secure. I feel a weird sense of victory over the elements."),
            DialogueTurn("t15", "The storm caused so much damage in the neighborhood. I saw an elderly neighbor struggling to clear their yard. I want to go help them. What tools should I bring?"),
            DialogueTurn("t16", "Give me a checklist of preventative landscaping maintenance I should schedule for the spring so this never happens again."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="home_repair_disaster_24_a",
        bucket="troubleshooting",
        title="Home Repair Disaster (24-Turn A)",
        system_preamble=(
            "You are an emergency support assistant helping the same user navigate an acute home crisis over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I just came downstairs and there is two inches of standing water in my basement. A pipe burst and I don't know what to do."),
            DialogueTurn("t2", "I found the main valve and shut the water off, but all my childhood photo albums were in cardboard boxes on the floor. They are soaked."),
            DialogueTurn("t3", "How do I even begin to dry out photos so they aren't destroyed forever?"),
            DialogueTurn("t4", "Should I call my homeowner's insurance right now, or wait until a plumber assesses the damage?"),
            DialogueTurn("t5", "My dog is barking aggressively at the water and driving me crazy. How do I get him to calm down?"),
            DialogueTurn("t6", "The plumber can't come until tomorrow morning. How do I prevent mold from starting overnight?"),
            DialogueTurn("t7", "Looking at all these ruined memories, I'm just sitting on the stairs crying. It feels so overwhelming."),
            DialogueTurn("t8", "Give me a clear, prioritized checklist of exactly what I need to do before going to sleep tonight."),
            DialogueTurn("t9", "The plumber arrived. The burst pipe is fixed, but the drywall is completely soaked. Do I have to tear it all out?"),
            DialogueTurn("t10", "The insurance adjuster said it's covered, but their preferred contractor can't start the mitigation for three days. Can I hire my own?"),
            DialogueTurn("t11", "I hired a local water damage team. They brought in massive industrial fans. The noise is deafening and giving me a headache. How long do these run?"),
            DialogueTurn("t12", "I spent the afternoon peeling apart the wet photos. I managed to save about half of them. The rest are destroyed. It hurts, but I feel less devastated than yesterday."),
            DialogueTurn("t13", "The mitigation team tore out the bottom two feet of drywall. My finished basement looks like a war zone. I hate living in construction."),
            DialogueTurn("t14", "Now I have to pick out new flooring. I want something completely waterproof this time. Is vinyl plank my best option?"),
            DialogueTurn("t15", "The insurance check finally cleared. I feel a massive sense of relief that I'm not paying for this out of pocket."),
            DialogueTurn("t16", "Help me draft a schedule to coordinate the flooring installers, the drywall team, and the painters over the next three weeks."),
            DialogueTurn("t17", "The drywallers are here, but they left a massive mess of dust all over the house. Is it my job to clean this or theirs?"),
            DialogueTurn("t18", "The flooring was installed today. The basement actually looks better than it did before the flood. I'm surprisingly happy."),
            DialogueTurn("t19", "I'm moving the furniture back in. To be safe, I bought heavy-duty plastic shelving for storage instead of cardboard boxes."),
            DialogueTurn("t20", "I bought water sensors that connect to my phone. How do I determine the best placement for them around the house?"),
            DialogueTurn("t21", "The whole project is officially done. The contractor handed over the final invoice. It feels like the end of a very stressful marathon."),
            DialogueTurn("t22", "I had my parents over to see the new basement. They were impressed. I feel like a much more capable homeowner now."),
            DialogueTurn("t23", "Every time it rains hard, I still get a twinge of panic and run to check the basement. Will that feeling ever fade?"),
            DialogueTurn("t24", "Give me a seasonal plumbing maintenance checklist so I can stay proactive and keep my house dry year-round."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="home_repair_disaster_24_b",
        bucket="troubleshooting",
        title="Home Repair Disaster (24-Turn B)",
        system_preamble=(
            "You are an emergency support assistant helping the same user navigate an acute home crisis over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "A massive tree branch just smashed through my living room window during a severe storm. Rain and wind are pouring into the house!"),
            DialogueTurn("t2", "We are all safe, but there is glass everywhere and the storm is still raging. How do I safely secure the room?"),
            DialogueTurn("t3", "I don't have plywood. I only have heavy duty trash bags and duct tape. Will that hold against the wind?"),
            DialogueTurn("t4", "The power just went out. It's freezing in here and pitch black. What is the protocol for this?"),
            DialogueTurn("t5", "I think the food in my fridge is going to spoil. Is it safe to put it in coolers out on the porch since it's cold outside?"),
            DialogueTurn("t6", "I called the insurance emergency line but I've been on hold for an hour. Should I just call a private tree removal service right now?"),
            DialogueTurn("t7", "My kids are terrified and crying because of the noise and the dark. How do I project calm when I'm internally panicking?"),
            DialogueTurn("t8", "Draft a quick list of what I need to photograph in the morning before we start cleaning up the debris."),
            DialogueTurn("t9", "It's morning. The storm passed. Seeing a tree trunk resting on my couch is surreal. The tree guys are coming soon. Do I touch anything?"),
            DialogueTurn("t10", "The tree is gone. The insurance adjuster gave us an emergency check for hotel stays because the house is too cold without the window. How long will this take?"),
            DialogueTurn("t11", "We are at a hotel. It feels like a weird, forced vacation. The kids love the pool, but I'm stressing about looters breaking into the boarded-up house."),
            DialogueTurn("t12", "I got a quote for a new custom window and it's going to take six weeks to manufacture. I cannot live in a hotel for six weeks."),
            DialogueTurn("t13", "A contractor offered to build a temporary insulated wall so we can move back in. Is that a common practice?"),
            DialogueTurn("t14", "The temporary wall is up. We moved back in. It's ugly, but the house is warm and secure. I feel a weird sense of victory over the elements."),
            DialogueTurn("t15", "The storm caused so much damage in the neighborhood. I saw an elderly neighbor struggling to clear their yard. I want to go help them. What tools should I bring?"),
            DialogueTurn("t16", "Give me a checklist of preventative landscaping maintenance I should schedule for the spring so this never happens again."),
            DialogueTurn("t17", "It's been five weeks. The window manufacturing was delayed by another two weeks. I am so tired of staring at an OSB wall in my living room."),
            DialogueTurn("t18", "The kids drew murals on the temporary wall. It actually looks kind of cool now. I guess we are making the best of it."),
            DialogueTurn("t19", "The window finally arrived! The installers are putting it in right now. I am irrationally excited to see natural light again."),
            DialogueTurn("t20", "It's in. The room feels whole again. We decided to buy a nicer couch to replace the crushed one. It feels like a fresh start."),
            DialogueTurn("t21", "The insurance company is fighting me on the cost of the replacement couch. How do I push back and prove its value?"),
            DialogueTurn("t22", "I submitted the original receipts and they approved the full amount. This whole ordeal has taught me to keep meticulous records."),
            DialogueTurn("t23", "We had a family movie night in the living room for the first time in two months. It felt so normal and so wonderful."),
            DialogueTurn("t24", "Give me a framework for an 'emergency readiness' family meeting so the kids know exactly what to do if the power goes out again."),
        ),
    ),

    # ==========================================
    # SKELETON 25: PUPPY ADOPTION JOY
    # ==========================================

    # LENGTH: 8 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="puppy_adoption_joy_8_a",
        bucket="pets_lifestyle",
        title="Puppy Adoption Joy (8-Turn A)",
        system_preamble=(
            "You are a dog training assistant helping the same user who just adopted a puppy over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I just brought home an 8-week-old Golden Retriever puppy! He is so sweet and goofy. Where do I even start with him tonight?"),
            DialogueTurn("t2", "I have a crate set up in the living room with a blanket. How do I make him like going in there?"),
            DialogueTurn("t3", "He's doing great, but he keeps trying to chew on my hands when we play. What's a good redirect?"),
            DialogueTurn("t4", "I want to make sure he's friendly with other dogs later. When can I start socializing him?"),
            DialogueTurn("t5", "Oh no, he just chewed the heel off one of my cheap flip-flops. It's my fault for leaving it out, but how do I teach 'leave it'?"),
            DialogueTurn("t6", "He seems to understand sit already! Are there any fun, easy tricks I can teach a puppy this young?"),
            DialogueTurn("t7", "I need to take him to the vet next week for his second round of shots. How do I make the car ride fun for him?"),
            DialogueTurn("t8", "Give me a fun, simple daily schedule for him including playtime, naps, and potty breaks."),
        ),
    ),

    # LENGTH: 8 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="puppy_adoption_joy_8_b",
        bucket="pets_lifestyle",
        title="Puppy Adoption Joy (8-Turn B)",
        system_preamble=(
            "You are a dog training assistant helping the same user who just adopted a puppy over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I just adopted a 10-week-old Australian Shepherd mix from the shelter! She is incredibly energetic and smart. What should I prioritize on day one?"),
            DialogueTurn("t2", "She seems scared of the stairs in my house. How do I help her build confidence without forcing her?"),
            DialogueTurn("t3", "She is a herding breed and keeps trying to nip at my ankles when I walk. How do I redirect that instinct?"),
            DialogueTurn("t4", "I bought a puzzle toy for her to eat her meals out of, but she figured it out in 30 seconds. How do I keep a smart dog entertained?"),
            DialogueTurn("t5", "My internet router just restarted randomly, but anyway, what kind of treats are best for high-value training?"),
            DialogueTurn("t6", "We mastered 'down' today! It only took 10 minutes. I feel like a proud parent. What's the next logical command to teach?"),
            DialogueTurn("t7", "I want to start leash training her in the backyard before we go on real walks. Any tips for a dog that pulls?"),
            DialogueTurn("t8", "Draft a checklist of essential puppy-proofing steps for my house so she stays safe while she explores."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="puppy_adoption_joy_16_a",
        bucket="pets_lifestyle",
        title="Puppy Adoption Joy (16-Turn A)",
        system_preamble=(
            "You are a dog training assistant helping the same user who just adopted a puppy over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I just brought home an 8-week-old Golden Retriever puppy! He is so sweet and goofy. Where do I even start with him tonight?"),
            DialogueTurn("t2", "I have a crate set up in the living room with a blanket. How do I make him like going in there?"),
            DialogueTurn("t3", "He's doing great, but he keeps trying to chew on my hands when we play. What's a good redirect?"),
            DialogueTurn("t4", "I want to make sure he's friendly with other dogs later. When can I start socializing him?"),
            DialogueTurn("t5", "Oh no, he just chewed the heel off one of my cheap flip-flops. It's my fault for leaving it out, but how do I teach 'leave it'?"),
            DialogueTurn("t6", "He seems to understand sit already! Are there any fun, easy tricks I can teach a puppy this young?"),
            DialogueTurn("t7", "I need to take him to the vet next week for his second round of shots. How do I make the car ride fun for him?"),
            DialogueTurn("t8", "Give me a fun, simple daily schedule for him including playtime, naps, and potty breaks."),
            DialogueTurn("t9", "We've had him for a month now. He sleeps through the night in his crate! I feel so rested. What's the next big milestone?"),
            DialogueTurn("t10", "He's starting to lose his puppy teeth. I keep finding tiny teeth on the carpet. What are the best toys to help his gums feel better?"),
            DialogueTurn("t11", "I took him to a puppy playgroup today. He was a little shy at first, but then he started wrestling with a doodle. It was so cute."),
            DialogueTurn("t12", "He's getting big fast. He learned how to counter-surf today and stole a piece of bread. How do I stop this before it becomes a habit?"),
            DialogueTurn("t13", "I used the 'place' command while I was cooking and he actually stayed on his mat. I am so proud of him."),
            DialogueTurn("t14", "We are going to a friend's house this weekend who has a cat. How do I introduce a goofy Golden to an older cat safely?"),
            DialogueTurn("t15", "The cat completely ignored him, and he just fell asleep on their rug. Best case scenario!"),
            DialogueTurn("t16", "Help me draft a list of fun socialization outings I can take him on this month to get him used to different environments."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="puppy_adoption_joy_16_b",
        bucket="pets_lifestyle",
        title="Puppy Adoption Joy (16-Turn B)",
        system_preamble=(
            "You are a dog training assistant helping the same user who just adopted a puppy over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I just adopted a 10-week-old Australian Shepherd mix from the shelter! She is incredibly energetic and smart. What should I prioritize on day one?"),
            DialogueTurn("t2", "She seems scared of the stairs in my house. How do I help her build confidence without forcing her?"),
            DialogueTurn("t3", "She is a herding breed and keeps trying to nip at my ankles when I walk. How do I redirect that instinct?"),
            DialogueTurn("t4", "I bought a puzzle toy for her to eat her meals out of, but she figured it out in 30 seconds. How do I keep a smart dog entertained?"),
            DialogueTurn("t5", "My internet router just restarted randomly, but anyway, what kind of treats are best for high-value training?"),
            DialogueTurn("t6", "We mastered 'down' today! It only took 10 minutes. I feel like a proud parent. What's the next logical command to teach?"),
            DialogueTurn("t7", "I want to start leash training her in the backyard before we go on real walks. Any tips for a dog that pulls?"),
            DialogueTurn("t8", "Draft a checklist of essential puppy-proofing steps for my house so she stays safe while she explores."),
            DialogueTurn("t9", "She's 4 months old now and entering the 'teenage' phase. She suddenly forgot how to come when called. Is she ignoring me on purpose?"),
            DialogueTurn("t10", "We went back to basics with a long lead. She did great! I also bought her a flirt pole. Herding dogs love these, right?"),
            DialogueTurn("t11", "She went crazy for the flirt pole. It tired her out in 15 minutes. This is a game-changer for rainy days."),
            DialogueTurn("t12", "I want to teach her to 'speak' or bark on command, but I don't want her to become a nuisance barker. Is there a trick to that?"),
            DialogueTurn("t13", "We learned 'speak' and 'quiet'! It's so fun to communicate with her. She is so eager to please."),
            DialogueTurn("t14", "I'm taking her on her first easy hiking trail tomorrow. Are there specific trail etiquette rules I need to know?"),
            DialogueTurn("t15", "The hike was magical. She stayed right by my side and ignored the squirrels. I think we are going to be best friends for a long time."),
            DialogueTurn("t16", "Give me a checklist of things to pack in a dedicated 'doggy hiking backpack' for future outdoor adventures."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="puppy_adoption_joy_24_a",
        bucket="pets_lifestyle",
        title="Puppy Adoption Joy (24-Turn A)",
        system_preamble=(
            "You are a dog training assistant helping the same user who just adopted a puppy over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I just brought home an 8-week-old Golden Retriever puppy! He is so sweet and goofy. Where do I even start with him tonight?"),
            DialogueTurn("t2", "I have a crate set up in the living room with a blanket. How do I make him like going in there?"),
            DialogueTurn("t3", "He's doing great, but he keeps trying to chew on my hands when we play. What's a good redirect?"),
            DialogueTurn("t4", "I want to make sure he's friendly with other dogs later. When can I start socializing him?"),
            DialogueTurn("t5", "Oh no, he just chewed the heel off one of my cheap flip-flops. It's my fault for leaving it out, but how do I teach 'leave it'?"),
            DialogueTurn("t6", "He seems to understand sit already! Are there any fun, easy tricks I can teach a puppy this young?"),
            DialogueTurn("t7", "I need to take him to the vet next week for his second round of shots. How do I make the car ride fun for him?"),
            DialogueTurn("t8", "Give me a fun, simple daily schedule for him including playtime, naps, and potty breaks."),
            DialogueTurn("t9", "We've had him for a month now. He sleeps through the night in his crate! I feel so rested. What's the next big milestone?"),
            DialogueTurn("t10", "He's starting to lose his puppy teeth. I keep finding tiny teeth on the carpet. What are the best toys to help his gums feel better?"),
            DialogueTurn("t11", "I took him to a puppy playgroup today. He was a little shy at first, but then he started wrestling with a doodle. It was so cute."),
            DialogueTurn("t12", "He's getting big fast. He learned how to counter-surf today and stole a piece of bread. How do I stop this before it becomes a habit?"),
            DialogueTurn("t13", "I used the 'place' command while I was cooking and he actually stayed on his mat. I am so proud of him."),
            DialogueTurn("t14", "We are going to a friend's house this weekend who has a cat. How do I introduce a goofy Golden to an older cat safely?"),
            DialogueTurn("t15", "The cat completely ignored him, and he just fell asleep on their rug. Best case scenario!"),
            DialogueTurn("t16", "Help me draft a list of fun socialization outings I can take him on this month to get him used to different environments."),
            DialogueTurn("t17", "He is officially six months old! He looks like a real dog now, not a potato. We are starting a group obedience class tonight."),
            DialogueTurn("t18", "In class, he was the class clown. He kept rolling on his back for belly rubs instead of heeling. How do I keep his focus around other dogs?"),
            DialogueTurn("t19", "I brought boiled chicken to the next class and he was the star pupil! High value treats really are magic."),
            DialogueTurn("t20", "We are taking him to the beach for the first time. Do Golden Retrievers naturally know how to swim, or do I need to teach him?"),
            DialogueTurn("t21", "He loved the water! He chased tennis balls into the waves all afternoon. Watching him run free makes me so happy."),
            DialogueTurn("t22", "I am bathing him to get the sand out, and he is being so patient in the tub. I really lucked out with his temperament."),
            DialogueTurn("t23", "He just fell asleep with his head on my lap. Bringing him home was the best decision I've made all year."),
            DialogueTurn("t24", "Give me a checklist for transitioning his diet and exercise routine as he moves from puppyhood into becoming an adult dog over the next few months."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="puppy_adoption_joy_24_b",
        bucket="pets_lifestyle",
        title="Puppy Adoption Joy (24-Turn B)",
        system_preamble=(
            "You are a dog training assistant helping the same user who just adopted a puppy over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I just adopted a 10-week-old Australian Shepherd mix from the shelter! She is incredibly energetic and smart. What should I prioritize on day one?"),
            DialogueTurn("t2", "She seems scared of the stairs in my house. How do I help her build confidence without forcing her?"),
            DialogueTurn("t3", "She is a herding breed and keeps trying to nip at my ankles when I walk. How do I redirect that instinct?"),
            DialogueTurn("t4", "I bought a puzzle toy for her to eat her meals out of, but she figured it out in 30 seconds. How do I keep a smart dog entertained?"),
            DialogueTurn("t5", "My internet router just restarted randomly, but anyway, what kind of treats are best for high-value training?"),
            DialogueTurn("t6", "We mastered 'down' today! It only took 10 minutes. I feel like a proud parent. What's the next logical command to teach?"),
            DialogueTurn("t7", "I want to start leash training her in the backyard before we go on real walks. Any tips for a dog that pulls?"),
            DialogueTurn("t8", "Draft a checklist of essential puppy-proofing steps for my house so she stays safe while she explores."),
            DialogueTurn("t9", "She's 4 months old now and entering the 'teenage' phase. She suddenly forgot how to come when called. Is she ignoring me on purpose?"),
            DialogueTurn("t10", "We went back to basics with a long lead. She did great! I also bought her a flirt pole. Herding dogs love these, right?"),
            DialogueTurn("t11", "She went crazy for the flirt pole. It tired her out in 15 minutes. This is a game-changer for rainy days."),
            DialogueTurn("t12", "I want to teach her to 'speak' or bark on command, but I don't want her to become a nuisance barker. Is there a trick to that?"),
            DialogueTurn("t13", "We learned 'speak' and 'quiet'! It's so fun to communicate with her. She is so eager to please."),
            DialogueTurn("t14", "I'm taking her on her first easy hiking trail tomorrow. Are there specific trail etiquette rules I need to know?"),
            DialogueTurn("t15", "The hike was magical. She stayed right by my side and ignored the squirrels. I think we are going to be best friends for a long time."),
            DialogueTurn("t16", "Give me a checklist of things to pack in a dedicated 'doggy hiking backpack' for future outdoor adventures."),
            DialogueTurn("t17", "She is eight months old now and almost fully grown. We want to try agility training just for fun. How do I make some DIY jumps in the yard?"),
            DialogueTurn("t18", "She loved the PVC pipe jumps! She cleared them instantly. I love watching her brain work out a problem."),
            DialogueTurn("t19", "I'm teaching her to weave through my legs as a party trick. She thinks it's hilarious. This dog brings so much joy into my house."),
            DialogueTurn("t20", "We are having a big family BBQ this weekend. She's friendly, but I don't want her jumping on kids. How do I set her up for success?"),
            DialogueTurn("t21", "The BBQ was great. I gave her a frozen Kong on her mat, and she just chilled while people ate. My family was so impressed with her behavior."),
            DialogueTurn("t22", "She lost her favorite squeaky toy under the couch and figured out how to use her paw to drag it out. She is too smart!"),
            DialogueTurn("t23", "I look at her sleeping on her bed and I can't imagine my life without her. The shelter really gave me the perfect dog."),
            DialogueTurn("t24", "Give me a framework for documenting her first year—maybe a scrapbook idea or a list of milestones we've hit together to celebrate her birthday next month."),
        ),
    ),

    # ==========================================
    # SKELETON 26: DREAM HONEYMOON PLANNING
    # ==========================================

    # LENGTH: 8 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="dream_honeymoon_planning_8_a",
        bucket="travel_leisure",
        title="Dream Honeymoon Planning (8-Turn A)",
        system_preamble=(
            "You are a travel advisor helping the same user plan a highly anticipated vacation over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "My fiancé and I are starting to plan our honeymoon to Japan for next spring! We are so excited. What are the absolute must-dos?"),
            DialogueTurn("t2", "We are huge foodies, so we want to prioritize amazing street food and maybe one fancy sushi dinner."),
            DialogueTurn("t3", "How does the bullet train system work? Do we need to buy passes before we fly there?"),
            DialogueTurn("t4", "We'd love to spend a night in a traditional ryokan with hot springs. Are those usually near Tokyo or further out?"),
            DialogueTurn("t5", "I just realized my passport expires three months after we get back. Will that be an issue at customs?"),
            DialogueTurn("t6", "We have about 14 days total. Should we try to squeeze in Kyoto and Osaka, or just stick to two cities?"),
            DialogueTurn("t7", "Neither of us speaks any Japanese, but we want to be polite. What are three phrases we absolutely must memorize?"),
            DialogueTurn("t8", "Help me draft a rough, fun 14-day itinerary balancing food, sights, and relaxation."),
        ),
    ),

    # LENGTH: 8 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="dream_honeymoon_planning_8_b",
        bucket="travel_leisure",
        title="Dream Honeymoon Planning (8-Turn B)",
        system_preamble=(
            "You are a travel advisor helping the same user plan a highly anticipated vacation over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "We just got married and we are planning our dream honeymoon to the Amalfi Coast in Italy for September! We want pure romance and pasta."),
            DialogueTurn("t2", "Should we fly into Rome and take a train down, or fly directly into Naples? We've never navigated European trains before."),
            DialogueTurn("t3", "We definitely want to do a boat tour of Capri. Is it better to rent a private boat or join a group tour to save some cash?"),
            DialogueTurn("t4", "I'm looking at hotels in Positano and the prices are wild. Is staying in a smaller town like Praiano just as beautiful?"),
            DialogueTurn("t5", "My work laptop just forced a bizarre software update, but anyway, do I need to pack an international driver's permit if we rent a Vespa?"),
            DialogueTurn("t6", "I think we'll skip the Vespa and stick to buses and ferries. I want to drink wine at lunch without worrying about cliffside driving."),
            DialogueTurn("t7", "We want to book one Michelin-starred dinner overlooking the water. Any advice on how far in advance we need reservations?"),
            DialogueTurn("t8", "Draft a 10-day romantic itinerary focused on relaxation, coastal views, and incredible dinners."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="dream_honeymoon_planning_16_a",
        bucket="travel_leisure",
        title="Dream Honeymoon Planning (16-Turn A)",
        system_preamble=(
            "You are a travel advisor helping the same user plan a highly anticipated vacation over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "My fiancé and I are starting to plan our honeymoon to Japan for next spring! We are so excited. What are the absolute must-dos?"),
            DialogueTurn("t2", "We are huge foodies, so we want to prioritize amazing street food and maybe one fancy sushi dinner."),
            DialogueTurn("t3", "How does the bullet train system work? Do we need to buy passes before we fly there?"),
            DialogueTurn("t4", "We'd love to spend a night in a traditional ryokan with hot springs. Are those usually near Tokyo or further out?"),
            DialogueTurn("t5", "I just realized my passport expires three months after we get back. Will that be an issue at customs?"),
            DialogueTurn("t6", "We have about 14 days total. Should we try to squeeze in Kyoto and Osaka, or just stick to two cities?"),
            DialogueTurn("t7", "Neither of us speaks any Japanese, but we want to be polite. What are three phrases we absolutely must memorize?"),
            DialogueTurn("t8", "Help me draft a rough, fun 14-day itinerary balancing food, sights, and relaxation."),
            DialogueTurn("t9", "We booked our flights! We fly into Tokyo and out of Osaka. It feels so real now!"),
            DialogueTurn("t10", "I'm looking into booking the ryokan in Hakone. Do they typically cater to dietary restrictions? My fiancé is allergic to shellfish."),
            DialogueTurn("t11", "I found a great ryokan that accommodates allergies. Now I'm looking at packing. What's the etiquette for clothing in temples and shrines?"),
            DialogueTurn("t12", "I'm so excited for the street food in Osaka. Is Dotonbori the best place to go, or is it a tourist trap?"),
            DialogueTurn("t13", "We want to buy some high-quality Japanese kitchen knives as a souvenir. Where is the best neighborhood in Tokyo for that?"),
            DialogueTurn("t14", "Our trip is next week! I'm starting to get nervous about the 14-hour flight. What are the best ways to combat jet lag once we land?"),
            DialogueTurn("t15", "We're mostly packed. I bought an e-SIM for data. I feel totally prepared and just giddy with excitement."),
            DialogueTurn("t16", "Draft a quick checklist of things to double-check in our carry-on bags before we head to the airport tomorrow morning."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="dream_honeymoon_planning_16_b",
        bucket="travel_leisure",
        title="Dream Honeymoon Planning (16-Turn B)",
        system_preamble=(
            "You are a travel advisor helping the same user plan a highly anticipated vacation over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "We just got married and we are planning our dream honeymoon to the Amalfi Coast in Italy for September! We want pure romance and pasta."),
            DialogueTurn("t2", "Should we fly into Rome and take a train down, or fly directly into Naples? We've never navigated European trains before."),
            DialogueTurn("t3", "We definitely want to do a boat tour of Capri. Is it better to rent a private boat or join a group tour to save some cash?"),
            DialogueTurn("t4", "I'm looking at hotels in Positano and the prices are wild. Is staying in a smaller town like Praiano just as beautiful?"),
            DialogueTurn("t5", "My work laptop just forced a bizarre software update, but anyway, do I need to pack an international driver's permit if we rent a Vespa?"),
            DialogueTurn("t6", "I think we'll skip the Vespa and stick to buses and ferries. I want to drink wine at lunch without worrying about cliffside driving."),
            DialogueTurn("t7", "We want to book one Michelin-starred dinner overlooking the water. Any advice on how far in advance we need reservations?"),
            DialogueTurn("t8", "Draft a 10-day romantic itinerary focused on relaxation, coastal views, and incredible dinners."),
            DialogueTurn("t9", "We booked a stunning Airbnb in Praiano! It has a balcony overlooking the sea. I can't wait to drink coffee out there."),
            DialogueTurn("t10", "Since we are staying in an Airbnb, we want to try cooking dinner ourselves one night. What's a classic, easy local dish we can buy ingredients for?"),
            DialogueTurn("t11", "I am looking into booking a lemon grove tour in Amalfi. Does that sound like a fun afternoon activity, or too touristy?"),
            DialogueTurn("t12", "We definitely want to do the 'Path of the Gods' hike. What kind of shoes should I pack? Do I need actual hiking boots?"),
            DialogueTurn("t13", "We are a month out! I'm trying to figure out cash versus cards. Do most restaurants and taxis there take Apple Pay now, or should I bring Euros?"),
            DialogueTurn("t14", "I'm so excited for the food. I'm going to eat gelato every single day. I'm not even thinking about my diet on this trip."),
            DialogueTurn("t15", "Our bags are packed! I can't believe it's finally here. All the wedding stress was worth it for this trip."),
            DialogueTurn("t16", "Draft a short 'out of office' email reply for me that playfully hints I'm eating my weight in pasta and won't be checking emails."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="dream_honeymoon_planning_24_a",
        bucket="travel_leisure",
        title="Dream Honeymoon Planning (24-Turn A)",
        system_preamble=(
            "You are a travel advisor helping the same user plan a highly anticipated vacation over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "My fiancé and I are starting to plan our honeymoon to Japan for next spring! We are so excited. What are the absolute must-dos?"),
            DialogueTurn("t2", "We are huge foodies, so we want to prioritize amazing street food and maybe one fancy sushi dinner."),
            DialogueTurn("t3", "How does the bullet train system work? Do we need to buy passes before we fly there?"),
            DialogueTurn("t4", "We'd love to spend a night in a traditional ryokan with hot springs. Are those usually near Tokyo or further out?"),
            DialogueTurn("t5", "I just realized my passport expires three months after we get back. Will that be an issue at customs?"),
            DialogueTurn("t6", "We have about 14 days total. Should we try to squeeze in Kyoto and Osaka, or just stick to two cities?"),
            DialogueTurn("t7", "Neither of us speaks any Japanese, but we want to be polite. What are three phrases we absolutely must memorize?"),
            DialogueTurn("t8", "Help me draft a rough, fun 14-day itinerary balancing food, sights, and relaxation."),
            DialogueTurn("t9", "We booked our flights! We fly into Tokyo and out of Osaka. It feels so real now!"),
            DialogueTurn("t10", "I'm looking into booking the ryokan in Hakone. Do they typically cater to dietary restrictions? My fiancé is allergic to shellfish."),
            DialogueTurn("t11", "I found a great ryokan that accommodates allergies. Now I'm looking at packing. What's the etiquette for clothing in temples and shrines?"),
            DialogueTurn("t12", "I'm so excited for the street food in Osaka. Is Dotonbori the best place to go, or is it a tourist trap?"),
            DialogueTurn("t13", "We want to buy some high-quality Japanese kitchen knives as a souvenir. Where is the best neighborhood in Tokyo for that?"),
            DialogueTurn("t14", "Our trip is next week! I'm starting to get nervous about the 14-hour flight. What are the best ways to combat jet lag once we land?"),
            DialogueTurn("t15", "We're mostly packed. I bought an e-SIM for data. I feel totally prepared and just giddy with excitement."),
            DialogueTurn("t16", "Draft a quick checklist of things to double-check in our carry-on bags before we head to the airport tomorrow morning."),
            DialogueTurn("t17", "We made it! We are in Tokyo. The jet lag is real, but grabbing a late-night snack from a 7-Eleven was actually magical."),
            DialogueTurn("t18", "We went to Kappabashi today and bought the most beautiful chef's knife. They even engraved our names on it. Best souvenir ever."),
            DialogueTurn("t19", "We are riding the bullet train to Kyoto right now. Watching the countryside zip by while eating an ekiben (bento box) is incredible."),
            DialogueTurn("t20", "The ryokan was the most relaxing experience of my life. The hot springs melted away all the lingering wedding stress."),
            DialogueTurn("t21", "We are in Osaka now, eating takoyaki by the canal. I never want to leave. The food here is on another level."),
            DialogueTurn("t22", "It's our last night. We are having a fancy wagyu beef dinner to celebrate. I feel so grateful for this trip."),
            DialogueTurn("t23", "We are at the airport flying home. We are exhausted but so happy. Japan exceeded every expectation."),
            DialogueTurn("t24", "Give me a fun framework for sorting through our 2,000 photos so we can make a beautiful, curated honeymoon photobook next week."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="dream_honeymoon_planning_24_b",
        bucket="travel_leisure",
        title="Dream Honeymoon Planning (24-Turn B)",
        system_preamble=(
            "You are a travel advisor helping the same user plan a highly anticipated vacation over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "We just got married and we are planning our dream honeymoon to the Amalfi Coast in Italy for September! We want pure romance and pasta."),
            DialogueTurn("t2", "Should we fly into Rome and take a train down, or fly directly into Naples? We've never navigated European trains before."),
            DialogueTurn("t3", "We definitely want to do a boat tour of Capri. Is it better to rent a private boat or join a group tour to save some cash?"),
            DialogueTurn("t4", "I'm looking at hotels in Positano and the prices are wild. Is staying in a smaller town like Praiano just as beautiful?"),
            DialogueTurn("t5", "My work laptop just forced a bizarre software update, but anyway, do I need to pack an international driver's permit if we rent a Vespa?"),
            DialogueTurn("t6", "I think we'll skip the Vespa and stick to buses and ferries. I want to drink wine at lunch without worrying about cliffside driving."),
            DialogueTurn("t7", "We want to book one Michelin-starred dinner overlooking the water. Any advice on how far in advance we need reservations?"),
            DialogueTurn("t8", "Draft a 10-day romantic itinerary focused on relaxation, coastal views, and incredible dinners."),
            DialogueTurn("t9", "We booked a stunning Airbnb in Praiano! It has a balcony overlooking the sea. I can't wait to drink coffee out there."),
            DialogueTurn("t10", "Since we are staying in an Airbnb, we want to try cooking dinner ourselves one night. What's a classic, easy local dish we can buy ingredients for?"),
            DialogueTurn("t11", "I am looking into booking a lemon grove tour in Amalfi. Does that sound like a fun afternoon activity, or too touristy?"),
            DialogueTurn("t12", "We definitely want to do the 'Path of the Gods' hike. What kind of shoes should I pack? Do I need actual hiking boots?"),
            DialogueTurn("t13", "We are a month out! I'm trying to figure out cash versus cards. Do most restaurants and taxis there take Apple Pay now, or should I bring Euros?"),
            DialogueTurn("t14", "I'm so excited for the food. I'm going to eat gelato every single day. I'm not even thinking about my diet on this trip."),
            DialogueTurn("t15", "Our bags are packed! I can't believe it's finally here. All the wedding stress was worth it for this trip."),
            DialogueTurn("t16", "Draft a short 'out of office' email reply for me that playfully hints I'm eating my weight in pasta and won't be checking emails."),
            DialogueTurn("t17", "We are here! We just checked into the Airbnb. The view is even better than the pictures. I am so overwhelmingly happy."),
            DialogueTurn("t18", "We went to a local market and bought fresh tomatoes, garlic, and pasta. We cooked dinner on our balcony tonight. Pure magic."),
            DialogueTurn("t19", "We did the boat tour to Capri today. Jumping into the Mediterranean off the side of a boat was the highlight of the trip so far."),
            DialogueTurn("t20", "We hiked the Path of the Gods. It was tough, but the views from the cliffs were breathtaking. We earned our pizza tonight."),
            DialogueTurn("t21", "We had our fancy Michelin-star dinner. The food was art, and watching the sunset over Positano was unforgettable."),
            DialogueTurn("t22", "It's our last day. We are just sitting at a cafe drinking spritzes and watching people walk by. I feel so deeply relaxed."),
            DialogueTurn("t23", "We are at the airport heading home. It was the perfect honeymoon. I wouldn't change a single thing."),
            DialogueTurn("t24", "Give me a checklist for 're-entry' back into normal life tomorrow so the post-vacation blues don't hit us too hard."),
        ),
    ),

    # ==========================================
    # SKELETON 27: GARAGE ORGANIZATION WEEKEND
    # ==========================================

    # LENGTH: 8 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="garage_organization_weekend_8_a",
        bucket="home_organization",
        title="Garage Organization Weekend (8-Turn A)",
        system_preamble=(
            "You are a home organization coach helping the same user tackle a mundane weekend project over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I have a totally free weekend and I'm finally going to clean out my messy two-car garage. What is the most efficient way to start?"),
            DialogueTurn("t2", "I have a lot of half-empty paint cans and old lawn chemicals. I know I can't just throw them in the trash."),
            DialogueTurn("t3", "I want to put up some wall shelving to get plastic bins off the floor. Do I need heavy-duty brackets for standard holiday decorations?"),
            DialogueTurn("t4", "Should I organize the bins by season, or by category like 'tools' vs 'sports'?"),
            DialogueTurn("t5", "I just found a box of my old high school yearbooks and got totally distracted looking through them for an hour."),
            DialogueTurn("t6", "Okay, back to work. I have three bicycles taking up too much room. What's the best way to hang them?"),
            DialogueTurn("t7", "I'm almost done and the floor is finally clear! Should I invest in an epoxy floor coating eventually, or is it not worth the money?"),
            DialogueTurn("t8", "Give me a checklist for a 10-minute monthly maintenance routine so the garage never gets this messy again."),
        ),
    ),

    # LENGTH: 8 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="garage_organization_weekend_8_b",
        bucket="home_organization",
        title="Garage Organization Weekend (8-Turn B)",
        system_preamble=(
            "You are a home organization coach helping the same user tackle a mundane weekend project over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I just inherited my grandfather's house, and the garage is packed to the ceiling with decades of clutter. I am completely overwhelmed. Where do I begin?"),
            DialogueTurn("t2", "I've started making 'keep', 'donate', and 'trash' piles, but I feel guilty throwing away his old rusty tools. What should I do with them?"),
            DialogueTurn("t3", "There is a massive oil stain on the concrete floor from his old truck. What's the best DIY way to lift that out?"),
            DialogueTurn("t4", "I found a box of old, unmarked keys. Should I just toss them, or keep them just in case?"),
            DialogueTurn("t5", "I got a splinter pulling apart an old workbench and had to stop to find a band-aid. The whole project feels cursed today."),
            DialogueTurn("t6", "I want to install a pegboard system for the tools I am keeping. Is it better to outline the tools with a marker so I know where they go?"),
            DialogueTurn("t7", "The overhead lighting in here is basically one dim yellow bulb. What's a cheap, easy lighting upgrade?"),
            DialogueTurn("t8", "Draft a prep-list for me so I can be ready to tackle the second half of the garage next weekend without losing my momentum."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="garage_organization_weekend_16_a",
        bucket="home_organization",
        title="Garage Organization Weekend (16-Turn A)",
        system_preamble=(
            "You are a home organization coach helping the same user tackle a mundane weekend project over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I have a totally free weekend and I'm finally going to clean out my messy two-car garage. What is the most efficient way to start?"),
            DialogueTurn("t2", "I have a lot of half-empty paint cans and old lawn chemicals. I know I can't just throw them in the trash."),
            DialogueTurn("t3", "I want to put up some wall shelving to get plastic bins off the floor. Do I need heavy-duty brackets for standard holiday decorations?"),
            DialogueTurn("t4", "Should I organize the bins by season, or by category like 'tools' vs 'sports'?"),
            DialogueTurn("t5", "I just found a box of my old high school yearbooks and got totally distracted looking through them for an hour."),
            DialogueTurn("t6", "Okay, back to work. I have three bicycles taking up too much room. What's the best way to hang them?"),
            DialogueTurn("t7", "I'm almost done and the floor is finally clear! Should I invest in an epoxy floor coating eventually, or is it not worth the money?"),
            DialogueTurn("t8", "Give me a checklist for a 10-minute monthly maintenance routine so the garage never gets this messy again."),
            DialogueTurn("t9", "I decided to actually do the epoxy floor! I bought a DIY kit. It says I have to acid-etch the concrete first. Is that dangerous?"),
            DialogueTurn("t10", "I'm wearing the protective gear and scrubbing. How do I know when the concrete is etched enough to accept the paint?"),
            DialogueTurn("t11", "I applied the base coat, but the fumes are incredibly strong. A neighbor actually complained. How do I ventilate this better?"),
            DialogueTurn("t12", "The smell dissipated. It's drying now. How long do I genuinely have to wait before I can drive my car onto it?"),
            DialogueTurn("t13", "It's fully cured! I'm moving the bins back in. But the labels I put on them are peeling off in the humidity. Any tricks for that?"),
            DialogueTurn("t14", "The garage looks like a showroom. It's so clean I kind of want to put a TV out here and make it a hangout space."),
            DialogueTurn("t15", "I'm having a beer in my pristine garage. It was a lot of hard work, but I feel incredibly satisfied."),
            DialogueTurn("t16", "Give me a list of three small, aesthetic upgrades (like matching storage buckets) to really finish off the 'showroom' look."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="garage_organization_weekend_16_b",
        bucket="home_organization",
        title="Garage Organization Weekend (16-Turn B)",
        system_preamble=(
            "You are a home organization coach helping the same user tackle a mundane weekend project over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I just inherited my grandfather's house, and the garage is packed to the ceiling with decades of clutter. I am completely overwhelmed. Where do I begin?"),
            DialogueTurn("t2", "I've started making 'keep', 'donate', and 'trash' piles, but I feel guilty throwing away his old rusty tools. What should I do with them?"),
            DialogueTurn("t3", "There is a massive oil stain on the concrete floor from his old truck. What's the best DIY way to lift that out?"),
            DialogueTurn("t4", "I found a box of old, unmarked keys. Should I just toss them, or keep them just in case?"),
            DialogueTurn("t5", "I got a splinter pulling apart an old workbench and had to stop to find a band-aid. The whole project feels cursed today."),
            DialogueTurn("t6", "I want to install a pegboard system for the tools I am keeping. Is it better to outline the tools with a marker so I know where they go?"),
            DialogueTurn("t7", "The overhead lighting in here is basically one dim yellow bulb. What's a cheap, easy lighting upgrade?"),
            DialogueTurn("t8", "Draft a prep-list for me so I can be ready to tackle the second half of the garage next weekend without losing my momentum."),
            DialogueTurn("t9", "I'm renting a dumpster for weekend two. What are the common things that dumpster rental companies absolutely will not let you throw away?"),
            DialogueTurn("t10", "While clearing the back corner, I found some mouse droppings. Do I need to call an exterminator or can I handle this myself?"),
            DialogueTurn("t11", "I set up some traps and cleaned the area with bleach. The clutter is finally gone! Now the bare walls look terrible. Should I paint them?"),
            DialogueTurn("t12", "I'm painting the walls a bright white. It instantly makes the space feel twice as large. I'm actually starting to enjoy this."),
            DialogueTurn("t13", "I want to build a simple, sturdy wooden workbench from scratch. What kind of wood should I buy at the hardware store for the top?"),
            DialogueTurn("t14", "I finished building the workbench! It's not perfectly level, but it's mine. I feel really proud of what I've accomplished here."),
            DialogueTurn("t15", "My grandfather would have loved this. It went from a hoarder's nightmare to a functional workspace."),
            DialogueTurn("t16", "Help me compile a checklist of essential starter tools I should buy to populate my new pegboard system."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="garage_organization_weekend_24_a",
        bucket="home_organization",
        title="Garage Organization Weekend (24-Turn A)",
        system_preamble=(
            "You are a home organization coach helping the same user tackle a mundane weekend project over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I have a totally free weekend and I'm finally going to clean out my messy two-car garage. What is the most efficient way to start?"),
            DialogueTurn("t2", "I have a lot of half-empty paint cans and old lawn chemicals. I know I can't just throw them in the trash."),
            DialogueTurn("t3", "I want to put up some wall shelving to get plastic bins off the floor. Do I need heavy-duty brackets for standard holiday decorations?"),
            DialogueTurn("t4", "Should I organize the bins by season, or by category like 'tools' vs 'sports'?"),
            DialogueTurn("t5", "I just found a box of my old high school yearbooks and got totally distracted looking through them for an hour."),
            DialogueTurn("t6", "Okay, back to work. I have three bicycles taking up too much room. What's the best way to hang them?"),
            DialogueTurn("t7", "I'm almost done and the floor is finally clear! Should I invest in an epoxy floor coating eventually, or is it not worth the money?"),
            DialogueTurn("t8", "Give me a checklist for a 10-minute monthly maintenance routine so the garage never gets this messy again."),
            DialogueTurn("t9", "I decided to actually do the epoxy floor! I bought a DIY kit. It says I have to acid-etch the concrete first. Is that dangerous?"),
            DialogueTurn("t10", "I'm wearing the protective gear and scrubbing. How do I know when the concrete is etched enough to accept the paint?"),
            DialogueTurn("t11", "I applied the base coat, but the fumes are incredibly strong. A neighbor actually complained. How do I ventilate this better?"),
            DialogueTurn("t12", "The smell dissipated. It's drying now. How long do I genuinely have to wait before I can drive my car onto it?"),
            DialogueTurn("t13", "It's fully cured! I'm moving the bins back in. But the labels I put on them are peeling off in the humidity. Any tricks for that?"),
            DialogueTurn("t14", "The garage looks like a showroom. It's so clean I kind of want to put a TV out here and make it a hangout space."),
            DialogueTurn("t15", "I'm having a beer in my pristine garage. It was a lot of hard work, but I feel incredibly satisfied."),
            DialogueTurn("t16", "Give me a list of three small, aesthetic upgrades (like matching storage buckets) to really finish off the 'showroom' look."),
            DialogueTurn("t17", "Winter is coming, and I want to use the garage as a home gym. But the garage door isn't insulated. Can I add insulation panels myself?"),
            DialogueTurn("t18", "I installed the foam panels. It's much warmer! But now I need a safe way to heat the space while I work out. Space heater?"),
            DialogueTurn("t19", "I've been working out in here all winter. It's great. But the snow melting off the car is creating puddles. How do I manage the water on my new epoxy floor?"),
            DialogueTurn("t20", "A squeegee did the trick. Spring is here, and it's time for my first 'spring cleaning' of the garage since the big overhaul. Where do I start?"),
            DialogueTurn("t21", "I want to host a garage sale to get rid of some of the stuff I organized into bins but haven't used all year. How do I price items quickly?"),
            DialogueTurn("t22", "The garage sale was a hit! I made $500 and cleared out three more bins. I love having an organized life."),
            DialogueTurn("t23", "Now that the garage is perfect, I'm eyeing the messy basement. Do you think the same organizational principles apply down there?"),
            DialogueTurn("t24", "Help me draft a project plan for tackling the basement, using everything I learned from the garage project."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="garage_organization_weekend_24_b",
        bucket="home_organization",
        title="Garage Organization Weekend (24-Turn B)",
        system_preamble=(
            "You are a home organization coach helping the same user tackle a mundane weekend project over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I just inherited my grandfather's house, and the garage is packed to the ceiling with decades of clutter. I am completely overwhelmed. Where do I begin?"),
            DialogueTurn("t2", "I've started making 'keep', 'donate', and 'trash' piles, but I feel guilty throwing away his old rusty tools. What should I do with them?"),
            DialogueTurn("t3", "There is a massive oil stain on the concrete floor from his old truck. What's the best DIY way to lift that out?"),
            DialogueTurn("t4", "I found a box of old, unmarked keys. Should I just toss them, or keep them just in case?"),
            DialogueTurn("t5", "I got a splinter pulling apart an old workbench and had to stop to find a band-aid. The whole project feels cursed today."),
            DialogueTurn("t6", "I want to install a pegboard system for the tools I am keeping. Is it better to outline the tools with a marker so I know where they go?"),
            DialogueTurn("t7", "The overhead lighting in here is basically one dim yellow bulb. What's a cheap, easy lighting upgrade?"),
            DialogueTurn("t8", "Draft a prep-list for me so I can be ready to tackle the second half of the garage next weekend without losing my momentum."),
            DialogueTurn("t9", "I'm renting a dumpster for weekend two. What are the common things that dumpster rental companies absolutely will not let you throw away?"),
            DialogueTurn("t10", "While clearing the back corner, I found some mouse droppings. Do I need to call an exterminator or can I handle this myself?"),
            DialogueTurn("t11", "I set up some traps and cleaned the area with bleach. The clutter is finally gone! Now the bare walls look terrible. Should I paint them?"),
            DialogueTurn("t12", "I'm painting the walls a bright white. It instantly makes the space feel twice as large. I'm actually starting to enjoy this."),
            DialogueTurn("t13", "I want to build a simple, sturdy wooden workbench from scratch. What kind of wood should I buy at the hardware store for the top?"),
            DialogueTurn("t14", "I finished building the workbench! It's not perfectly level, but it's mine. I feel really proud of what I've accomplished here."),
            DialogueTurn("t15", "My grandfather would have loved this. It went from a hoarder's nightmare to a functional workspace."),
            DialogueTurn("t16", "Help me compile a checklist of essential starter tools I should buy to populate my new pegboard system."),
            DialogueTurn("t17", "I'm starting to get into woodworking in the new garage. But the sawdust is getting everywhere. What's a beginner-friendly dust collection setup?"),
            DialogueTurn("t18", "I hooked up a shop vac to a cyclone separator. It works great! But my power tools are really loud. I'm worried about annoying the neighbors."),
            DialogueTurn("t19", "Can I add soundproofing foam to the garage walls, or is that a fire hazard in a workshop?"),
            DialogueTurn("t20", "I'm building my first real project: a birdhouse. I keep messing up the angled cuts on my miter saw. Any tips for precision?"),
            DialogueTurn("t21", "The birdhouse is done and painted. I hung it in the backyard. This hobby is so incredibly rewarding compared to my desk job."),
            DialogueTurn("t22", "I want to start selling small wooden crafts at the local market. Is it easy to transition this from a hobby into a small side hustle?"),
            DialogueTurn("t23", "I think I'll keep it as a hobby for now. I don't want to ruin the peace of the workshop with deadlines and stress."),
            DialogueTurn("t24", "Give me a safety checklist I should review every time I step into the garage to use the power saws, so I never get complacent."),
        ),
    ),

    # ==========================================
    # SKELETON 28: CASUAL INDOOR GARDENING
    # ==========================================

    # LENGTH: 8 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="casual_indoor_gardening_8_a",
        bucket="hobbies",
        title="Casual Indoor Gardening (8-Turn A)",
        system_preamble=(
            "You are a botany assistant helping the same user start a relaxing new hobby over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I want to add some greenery to my apartment, but I've never kept a plant alive before. What are the best beginner houseplants?"),
            DialogueTurn("t2", "My living room only has one north-facing window, so it doesn't get very bright in there."),
            DialogueTurn("t3", "I bought a Pothos and a Snake Plant! Do I need to repot them immediately, or keep them in the plastic nursery pots?"),
            DialogueTurn("t4", "How do I know when to water them? I'm terrified of overwatering."),
            DialogueTurn("t5", "My friend is bringing her cat over this weekend. Are either of these plants toxic to pets?"),
            DialogueTurn("t6", "I saw people online wiping the dust off their plant leaves. Is that actually necessary, or just for aesthetics?"),
            DialogueTurn("t7", "The Pothos is actually growing a new leaf! It's so satisfying to watch. Do they need fertilizer in the winter?"),
            DialogueTurn("t8", "Give me a simple, printable watering and care schedule for these two specific plants."),
        ),
    ),

    # LENGTH: 8 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="casual_indoor_gardening_8_b",
        bucket="hobbies",
        title="Casual Indoor Gardening (8-Turn B)",
        system_preamble=(
            "You are a botany assistant helping the same user start a relaxing new hobby over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I love the look of succulents, but I always end up killing them. They just turn to mush. What am I doing wrong?"),
            DialogueTurn("t2", "I bought some new ones and got terracotta pots with drainage holes this time. What kind of soil should I buy for them?"),
            DialogueTurn("t3", "I live in a basement apartment with very little natural light. Can I use a cheap LED desk lamp as a grow light?"),
            DialogueTurn("t4", "I accidentally knocked one over and spilled dirt all over my white rug. Is there a trick to vacuuming potting soil without staining the rug?"),
            DialogueTurn("t5", "Two of the leaves broke off when it fell. Can I grow new plants from those broken leaves?"),
            DialogueTurn("t6", "I've been letting the leaves callous over like you said. Now I see tiny pink roots! It's like magic. Do I bury them now?"),
            DialogueTurn("t7", "I noticed tiny little black flies buzzing around the soil today. Are those dangerous to the plant?"),
            DialogueTurn("t8", "Draft a shopping list for me of three things I need to buy at the hardware store to deal with these fungus gnats safely."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="casual_indoor_gardening_16_a",
        bucket="hobbies",
        title="Casual Indoor Gardening (16-Turn A)",
        system_preamble=(
            "You are a botany assistant helping the same user start a relaxing new hobby over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I want to add some greenery to my apartment, but I've never kept a plant alive before. What are the best beginner houseplants?"),
            DialogueTurn("t2", "My living room only has one north-facing window, so it doesn't get very bright in there."),
            DialogueTurn("t3", "I bought a Pothos and a Snake Plant! Do I need to repot them immediately, or keep them in the plastic nursery pots?"),
            DialogueTurn("t4", "How do I know when to water them? I'm terrified of overwatering."),
            DialogueTurn("t5", "My friend is bringing her cat over this weekend. Are either of these plants toxic to pets?"),
            DialogueTurn("t6", "I saw people online wiping the dust off their plant leaves. Is that actually necessary, or just for aesthetics?"),
            DialogueTurn("t7", "The Pothos is actually growing a new leaf! It's so satisfying to watch. Do they need fertilizer in the winter?"),
            DialogueTurn("t8", "Give me a simple, printable watering and care schedule for these two specific plants."),
            DialogueTurn("t9", "It's been six months. The Pothos is getting incredibly long and trailing on the floor. Should I trim it?"),
            DialogueTurn("t10", "I want to try propagating the pieces I trimmed off. Should I put them in a glass of water or directly into wet soil?"),
            DialogueTurn("t11", "I put them in water on my windowsill. It's been two weeks and I see white roots growing! This is the coolest hobby."),
            DialogueTurn("t12", "I planted the rooted cuttings in a cute mug, but I just realized it doesn't have a drainage hole. Will it die?"),
            DialogueTurn("t13", "I carefully drilled a hole in the bottom of the mug! Now I want to gift this baby plant to a coworker. How should I care for it before gifting?"),
            DialogueTurn("t14", "I feel confident now. I want to buy a big, statement plant. Is a Monstera Deliciosa too hard for a beginner?"),
            DialogueTurn("t15", "I bought a gorgeous Monstera. It has those beautiful splits in the leaves. I am officially a plant person."),
            DialogueTurn("t16", "Help me draft a quick list of signs I should look for that indicate my new Monstera is unhappy in its new environment."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="casual_indoor_gardening_16_b",
        bucket="hobbies",
        title="Casual Indoor Gardening (16-Turn B)",
        system_preamble=(
            "You are a botany assistant helping the same user start a relaxing new hobby over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I love the look of succulents, but I always end up killing them. They just turn to mush. What am I doing wrong?"),
            DialogueTurn("t2", "I bought some new ones and got terracotta pots with drainage holes this time. What kind of soil should I buy for them?"),
            DialogueTurn("t3", "I live in a basement apartment with very little natural light. Can I use a cheap LED desk lamp as a grow light?"),
            DialogueTurn("t4", "I accidentally knocked one over and spilled dirt all over my white rug. Is there a trick to vacuuming potting soil without staining the rug?"),
            DialogueTurn("t5", "Two of the leaves broke off when it fell. Can I grow new plants from those broken leaves?"),
            DialogueTurn("t6", "I've been letting the leaves callous over like you said. Now I see tiny pink roots! It's like magic. Do I bury them now?"),
            DialogueTurn("t7", "I noticed tiny little black flies buzzing around the soil today. Are those dangerous to the plant?"),
            DialogueTurn("t8", "Draft a shopping list for me of three things I need to buy at the hardware store to deal with these fungus gnats safely."),
            DialogueTurn("t9", "The neem oil and sticky traps worked! The gnats are gone. But now the bottom leaves of my Echeveria are turning yellow and falling off."),
            DialogueTurn("t10", "I've been watering it from the top, but I read about 'bottom watering'. What is that, and is it better for succulents?"),
            DialogueTurn("t11", "I tried bottom watering today. It's so cool watching the soil soak it up. I feel like a plant scientist."),
            DialogueTurn("t12", "I'm going on a two-week vacation soon. Do I need to hire a plant sitter for my succulents?"),
            DialogueTurn("t13", "I got back from vacation. The succulents didn't even notice I was gone. Now I want to expand my collection. Is a ZZ plant as indestructible as people say?"),
            DialogueTurn("t14", "I bought a ZZ plant! It's so dark green and shiny. Does it need a different type of fertilizer than the succulents?"),
            DialogueTurn("t15", "Taking care of these plants has actually become a really grounding morning ritual for me. It lowers my stress before work."),
            DialogueTurn("t16", "Give me a simple guide on how to safely fertilize both succulents and a ZZ plant during the active growing season."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="casual_indoor_gardening_24_a",
        bucket="hobbies",
        title="Casual Indoor Gardening (24-Turn A)",
        system_preamble=(
            "You are a botany assistant helping the same user start a relaxing new hobby over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I want to add some greenery to my apartment, but I've never kept a plant alive before. What are the best beginner houseplants?"),
            DialogueTurn("t2", "My living room only has one north-facing window, so it doesn't get very bright in there."),
            DialogueTurn("t3", "I bought a Pothos and a Snake Plant! Do I need to repot them immediately, or keep them in the plastic nursery pots?"),
            DialogueTurn("t4", "How do I know when to water them? I'm terrified of overwatering."),
            DialogueTurn("t5", "My friend is bringing her cat over this weekend. Are either of these plants toxic to pets?"),
            DialogueTurn("t6", "I saw people online wiping the dust off their plant leaves. Is that actually necessary, or just for aesthetics?"),
            DialogueTurn("t7", "The Pothos is actually growing a new leaf! It's so satisfying to watch. Do they need fertilizer in the winter?"),
            DialogueTurn("t8", "Give me a simple, printable watering and care schedule for these two specific plants."),
            DialogueTurn("t9", "It's been six months. The Pothos is getting incredibly long and trailing on the floor. Should I trim it?"),
            DialogueTurn("t10", "I want to try propagating the pieces I trimmed off. Should I put them in a glass of water or directly into wet soil?"),
            DialogueTurn("t11", "I put them in water on my windowsill. It's been two weeks and I see white roots growing! This is the coolest hobby."),
            DialogueTurn("t12", "I planted the rooted cuttings in a cute mug, but I just realized it doesn't have a drainage hole. Will it die?"),
            DialogueTurn("t13", "I carefully drilled a hole in the bottom of the mug! Now I want to gift this baby plant to a coworker. How should I care for it before gifting?"),
            DialogueTurn("t14", "I feel confident now. I want to buy a big, statement plant. Is a Monstera Deliciosa too hard for a beginner?"),
            DialogueTurn("t15", "I bought a gorgeous Monstera. It has those beautiful splits in the leaves. I am officially a plant person."),
            DialogueTurn("t16", "Help me draft a quick list of signs I should look for that indicate my new Monstera is unhappy in its new environment."),
            DialogueTurn("t17", "Disaster! I found tiny little webs on the back of the Monstera leaves. Are these spider mites? I'm panicking."),
            DialogueTurn("t18", "I immediately moved it to the bathroom away from my other plants. How do I treat it without using harsh chemicals in my apartment?"),
            DialogueTurn("t19", "I sprayed it down with the soap and water mixture. It took an hour. Will I have to do this every day?"),
            DialogueTurn("t20", "It's been three weeks. The mites are gone and the Monstera just pushed out a massive new leaf! I saved it!"),
            DialogueTurn("t21", "The Monstera is getting so heavy it's starting to lean over. Someone suggested a 'moss pole'. What is that?"),
            DialogueTurn("t22", "I installed the moss pole and gently tied the stems to it. It looks so much more structural and healthy now."),
            DialogueTurn("t23", "I have 12 plants in my apartment now. It feels like a jungle, but in the best way possible. It's my happy place."),
            DialogueTurn("t24", "Give me a checklist for a comprehensive 'spring cleaning' routine for my entire plant collection so they thrive this summer."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="casual_indoor_gardening_24_b",
        bucket="hobbies",
        title="Casual Indoor Gardening (24-Turn B)",
        system_preamble=(
            "You are a botany assistant helping the same user start a relaxing new hobby over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I love the look of succulents, but I always end up killing them. They just turn to mush. What am I doing wrong?"),
            DialogueTurn("t2", "I bought some new ones and got terracotta pots with drainage holes this time. What kind of soil should I buy for them?"),
            DialogueTurn("t3", "I live in a basement apartment with very little natural light. Can I use a cheap LED desk lamp as a grow light?"),
            DialogueTurn("t4", "I accidentally knocked one over and spilled dirt all over my white rug. Is there a trick to vacuuming potting soil without staining the rug?"),
            DialogueTurn("t5", "Two of the leaves broke off when it fell. Can I grow new plants from those broken leaves?"),
            DialogueTurn("t6", "I've been letting the leaves callous over like you said. Now I see tiny pink roots! It's like magic. Do I bury them now?"),
            DialogueTurn("t7", "I noticed tiny little black flies buzzing around the soil today. Are those dangerous to the plant?"),
            DialogueTurn("t8", "Draft a shopping list for me of three things I need to buy at the hardware store to deal with these fungus gnats safely."),
            DialogueTurn("t9", "The neem oil and sticky traps worked! The gnats are gone. But now the bottom leaves of my Echeveria are turning yellow and falling off."),
            DialogueTurn("t10", "I've been watering it from the top, but I read about 'bottom watering'. What is that, and is it better for succulents?"),
            DialogueTurn("t11", "I tried bottom watering today. It's so cool watching the soil soak it up. I feel like a plant scientist."),
            DialogueTurn("t12", "I'm going on a two-week vacation soon. Do I need to hire a plant sitter for my succulents?"),
            DialogueTurn("t13", "I got back from vacation. The succulents didn't even notice I was gone. Now I want to expand my collection. Is a ZZ plant as indestructible as people say?"),
            DialogueTurn("t14", "I bought a ZZ plant! It's so dark green and shiny. Does it need a different type of fertilizer than the succulents?"),
            DialogueTurn("t15", "Taking care of these plants has actually become a really grounding morning ritual for me. It lowers my stress before work."),
            DialogueTurn("t16", "Give me a simple guide on how to safely fertilize both succulents and a ZZ plant during the active growing season."),
            DialogueTurn("t17", "Winter is here. My apartment gets really cold and drafty near the windows. Should I move all my plants away from the glass?"),
            DialogueTurn("t18", "I moved them. The ZZ plant hasn't grown a single leaf in three months. Is it dead, or just asleep?"),
            DialogueTurn("t19", "Spring has arrived! I see new growth. But I think they might be 'root bound'. How can I tell if a plant needs a bigger pot?"),
            DialogueTurn("t20", "I pulled the ZZ plant out of the pot and it's literally all thick, potato-like roots. It definitely needs a bigger home. Is plastic or ceramic better?"),
            DialogueTurn("t21", "I repotted it into a beautiful glazed ceramic pot. Now I want to try growing something I can actually eat. Are indoor herbs difficult?"),
            DialogueTurn("t22", "I bought a basil plant for my kitchen counter. It smells amazing. How do I harvest leaves without stunting its growth?"),
            DialogueTurn("t23", "I made my first batch of pasta using basil I grew myself. It tasted infinitely better because I grew it. I love this hobby."),
            DialogueTurn("t24", "Give me a brainstorming list of five other easy, culinary herbs I can grow in small pots on my kitchen windowsill this summer."),
        ),
    ),

    # ==========================================
    # SKELETON 29: FIRST DAY NEW JOB
    # ==========================================

    # LENGTH: 8 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="first_day_new_job_8_a",
        bucket="career_positive",
        title="First Day New Job (8-Turn A)",
        system_preamble=(
            "You are a career mentor helping the same user prepare for a positive professional milestone over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I start my new job as a marketing manager on Monday! I'm not nervous, just really eager to make a great first impression. Any tips?"),
            DialogueTurn("t2", "The dress code is 'smart casual', which is pretty vague. Should I lean more towards a blazer or a nice sweater for day one?"),
            DialogueTurn("t3", "I'm going to be introduced to a lot of people. What's a good, natural elevator pitch for myself?"),
            DialogueTurn("t4", "I want to take notes during onboarding, but I don't want to look like I'm hiding behind my laptop. Is a physical notebook better?"),
            DialogueTurn("t5", "I was practicing my commute this morning and spilled coffee on my passenger seat. What's the best way to get that out of upholstery?"),
            DialogueTurn("t6", "If my manager asks me to lunch, should I offer to pay for my own, or assume they are expensing it?"),
            DialogueTurn("t7", "I want to establish good boundaries early on without seeming unhelpful. How do I leave on time my first week?"),
            DialogueTurn("t8", "Give me a checklist of three realistic goals to accomplish by the end of my very first Friday."),
        ),
    ),

    # LENGTH: 8 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="first_day_new_job_8_b",
        bucket="career_positive",
        title="First Day New Job (8-Turn B)",
        system_preamble=(
            "You are a career mentor helping the same user prepare for a positive professional milestone over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I landed my first fully remote job as a junior software developer! I start tomorrow and I'm so pumped. How do I prepare my home office today?"),
            DialogueTurn("t2", "I have a video call with the whole engineering team tomorrow morning. Should I blur my background or just make sure my room is clean?"),
            DialogueTurn("t3", "They use Slack heavily. What is the general etiquette for introducing yourself in a large, remote company channel?"),
            DialogueTurn("t4", "I'm worried about 'Zoom fatigue' since I have 6 hours of onboarding calls scheduled. How do I stay energized on camera?"),
            DialogueTurn("t5", "My dog barks at the mailman every day at 11 AM. What's a polite way to mute or handle that if it happens during a meeting?"),
            DialogueTurn("t6", "I just read through their codebase and I feel a sudden wave of imposter syndrome. It's so much more complex than my bootcamp projects."),
            DialogueTurn("t7", "I have a 1-on-1 with my direct manager on Friday. What should my primary goal be for that specific meeting?"),
            DialogueTurn("t8", "Draft a template for a 'Friday wrap-up' message I can send my manager to show them I had a productive first week."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="first_day_new_job_16_a",
        bucket="career_positive",
        title="First Day New Job (16-Turn A)",
        system_preamble=(
            "You are a career mentor helping the same user prepare for a positive professional milestone over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I start my new job as a marketing manager on Monday! I'm not nervous, just really eager to make a great first impression. Any tips?"),
            DialogueTurn("t2", "The dress code is 'smart casual', which is pretty vague. Should I lean more towards a blazer or a nice sweater for day one?"),
            DialogueTurn("t3", "I'm going to be introduced to a lot of people. What's a good, natural elevator pitch for myself?"),
            DialogueTurn("t4", "I want to take notes during onboarding, but I don't want to look like I'm hiding behind my laptop. Is a physical notebook better?"),
            DialogueTurn("t5", "I was practicing my commute this morning and spilled coffee on my passenger seat. What's the best way to get that out of upholstery?"),
            DialogueTurn("t6", "If my manager asks me to lunch, should I offer to pay for my own, or assume they are expensing it?"),
            DialogueTurn("t7", "I want to establish good boundaries early on without seeming unhelpful. How do I leave on time my first week?"),
            DialogueTurn("t8", "Give me a checklist of three realistic goals to accomplish by the end of my very first Friday."),
            DialogueTurn("t9", "Week one was great! Now it's week two and I've been assigned my first major project. I have to present a campaign idea to the sales team."),
            DialogueTurn("t10", "In a planning meeting, a senior sales rep completely dismissed my idea. I stayed quiet. How should I handle pushback politely next time?"),
            DialogueTurn("t11", "I used your advice and clarified my data with the rep offline. They actually agreed with me! It felt like a massive win."),
            DialogueTurn("t12", "I'm meeting the VP of Marketing tomorrow. I'm suddenly intimidated. How do I talk strategy with an executive without sounding out of my depth?"),
            DialogueTurn("t13", "The meeting with the VP went amazingly well. They praised my campaign idea. I am officially feeling confident in this role."),
            DialogueTurn("t14", "I have to file my first expense report for a client dinner. I lost one of the $15 receipts. Is it better to just pay out of pocket or ask finance what to do?"),
            DialogueTurn("t15", "Finance was super chill about it. The company culture here is so much healthier than my last job. I'm really happy."),
            DialogueTurn("t16", "Draft a short self-evaluation outline I can use to prepare for my official 30-day check-in with my manager next week."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="first_day_new_job_16_b",
        bucket="career_positive",
        title="First Day New Job (16-Turn B)",
        system_preamble=(
            "You are a career mentor helping the same user prepare for a positive professional milestone over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I landed my first fully remote job as a junior software developer! I start tomorrow and I'm so pumped. How do I prepare my home office today?"),
            DialogueTurn("t2", "I have a video call with the whole engineering team tomorrow morning. Should I blur my background or just make sure my room is clean?"),
            DialogueTurn("t3", "They use Slack heavily. What is the general etiquette for introducing yourself in a large, remote company channel?"),
            DialogueTurn("t4", "I'm worried about 'Zoom fatigue' since I have 6 hours of onboarding calls scheduled. How do I stay energized on camera?"),
            DialogueTurn("t5", "My dog barks at the mailman every day at 11 AM. What's a polite way to mute or handle that if it happens during a meeting?"),
            DialogueTurn("t6", "I just read through their codebase and I feel a sudden wave of imposter syndrome. It's so much more complex than my bootcamp projects."),
            DialogueTurn("t7", "I have a 1-on-1 with my direct manager on Friday. What should my primary goal be for that specific meeting?"),
            DialogueTurn("t8", "Draft a template for a 'Friday wrap-up' message I can send my manager to show them I had a productive first week."),
            DialogueTurn("t9", "Week two begins! I am supposed to pair-program with a senior developer today. I am terrified of typing slow or making a stupid mistake while they watch."),
            DialogueTurn("t10", "The pair programming was actually awesome. They were so patient. I'm realizing remote work requires a lot more proactive communication. How often should I update my team?"),
            DialogueTurn("t11", "I have colleagues in London and India. I'm in New York. How do I navigate timezone delays without being a bottleneck?"),
            DialogueTurn("t12", "I am submitting my very first Pull Request (PR) today. What is the best way to write a description so the reviewers don't hate me?"),
            DialogueTurn("t13", "My PR got five comments requesting changes. I feel a bit defensive, even though I know it's normal. How do I process code reviews objectively?"),
            DialogueTurn("t14", "I made the changes, and they approved and merged it! My code is officially in production. I am celebrating in my living room!"),
            DialogueTurn("t15", "I notice a lot of institutional knowledge isn't documented anywhere. Should I start writing documentation as a junior dev, or is that overstepping?"),
            DialogueTurn("t16", "Help me draft a polite message proposing a new 'Developer Wiki' page to my manager to show initiative."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="first_day_new_job_24_a",
        bucket="career_positive",
        title="First Day New Job (24-Turn A)",
        system_preamble=(
            "You are a career mentor helping the same user prepare for a positive professional milestone over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I start my new job as a marketing manager on Monday! I'm not nervous, just really eager to make a great first impression. Any tips?"),
            DialogueTurn("t2", "The dress code is 'smart casual', which is pretty vague. Should I lean more towards a blazer or a nice sweater for day one?"),
            DialogueTurn("t3", "I'm going to be introduced to a lot of people. What's a good, natural elevator pitch for myself?"),
            DialogueTurn("t4", "I want to take notes during onboarding, but I don't want to look like I'm hiding behind my laptop. Is a physical notebook better?"),
            DialogueTurn("t5", "I was practicing my commute this morning and spilled coffee on my passenger seat. What's the best way to get that out of upholstery?"),
            DialogueTurn("t6", "If my manager asks me to lunch, should I offer to pay for my own, or assume they are expensing it?"),
            DialogueTurn("t7", "I want to establish good boundaries early on without seeming unhelpful. How do I leave on time my first week?"),
            DialogueTurn("t8", "Give me a checklist of three realistic goals to accomplish by the end of my very first Friday."),
            DialogueTurn("t9", "Week one was great! Now it's week two and I've been assigned my first major project. I have to present a campaign idea to the sales team."),
            DialogueTurn("t10", "In a planning meeting, a senior sales rep completely dismissed my idea. I stayed quiet. How should I handle pushback politely next time?"),
            DialogueTurn("t11", "I used your advice and clarified my data with the rep offline. They actually agreed with me! It felt like a massive win."),
            DialogueTurn("t12", "I'm meeting the VP of Marketing tomorrow. I'm suddenly intimidated. How do I talk strategy with an executive without sounding out of my depth?"),
            DialogueTurn("t13", "The meeting with the VP went amazingly well. They praised my campaign idea. I am officially feeling confident in this role."),
            DialogueTurn("t14", "I have to file my first expense report for a client dinner. I lost one of the $15 receipts. Is it better to just pay out of pocket or ask finance what to do?"),
            DialogueTurn("t15", "Finance was super chill about it. The company culture here is so much healthier than my last job. I'm really happy."),
            DialogueTurn("t16", "Draft a short self-evaluation outline I can use to prepare for my official 30-day check-in with my manager next week."),
            DialogueTurn("t17", "My 30-day review was glowing. They are actually assigning me a junior coordinator to manage. I've never managed a direct report before!"),
            DialogueTurn("t18", "How do I balance being a friendly, approachable manager with being an authoritative boss who holds them accountable?"),
            DialogueTurn("t19", "My new direct report is eager but keeps making small detail errors. How do I give constructive feedback without crushing their spirit?"),
            DialogueTurn("t20", "The feedback sandwich method worked perfectly. They improved immediately. I'm really enjoying the mentorship aspect of this job."),
            DialogueTurn("t21", "It's time for annual performance reviews. I want to ask for a raise because I've taken on management duties. How do I build a solid case?"),
            DialogueTurn("t22", "I presented my case. They didn't just give me a raise, they bumped me up to Senior Marketing Manager! I am thrilled."),
            DialogueTurn("t23", "I took my team out for drinks to celebrate our department's success this year. I feel so deeply fulfilled by my career right now."),
            DialogueTurn("t24", "Give me a framework for setting aggressive but healthy professional development goals for my second year at the company."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="first_day_new_job_24_b",
        bucket="career_positive",
        title="First Day New Job (24-Turn B)",
        system_preamble=(
            "You are a career mentor helping the same user prepare for a positive professional milestone over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I landed my first fully remote job as a junior software developer! I start tomorrow and I'm so pumped. How do I prepare my home office today?"),
            DialogueTurn("t2", "I have a video call with the whole engineering team tomorrow morning. Should I blur my background or just make sure my room is clean?"),
            DialogueTurn("t3", "They use Slack heavily. What is the general etiquette for introducing yourself in a large, remote company channel?"),
            DialogueTurn("t4", "I'm worried about 'Zoom fatigue' since I have 6 hours of onboarding calls scheduled. How do I stay energized on camera?"),
            DialogueTurn("t5", "My dog barks at the mailman every day at 11 AM. What's a polite way to mute or handle that if it happens during a meeting?"),
            DialogueTurn("t6", "I just read through their codebase and I feel a sudden wave of imposter syndrome. It's so much more complex than my bootcamp projects."),
            DialogueTurn("t7", "I have a 1-on-1 with my direct manager on Friday. What should my primary goal be for that specific meeting?"),
            DialogueTurn("t8", "Draft a template for a 'Friday wrap-up' message I can send my manager to show them I had a productive first week."),
            DialogueTurn("t9", "Week two begins! I am supposed to pair-program with a senior developer today. I am terrified of typing slow or making a stupid mistake while they watch."),
            DialogueTurn("t10", "The pair programming was actually awesome. They were so patient. I'm realizing remote work requires a lot more proactive communication. How often should I update my team?"),
            DialogueTurn("t11", "I have colleagues in London and India. I'm in New York. How do I navigate timezone delays without being a bottleneck?"),
            DialogueTurn("t12", "I am submitting my very first Pull Request (PR) today. What is the best way to write a description so the reviewers don't hate me?"),
            DialogueTurn("t13", "My PR got five comments requesting changes. I feel a bit defensive, even though I know it's normal. How do I process code reviews objectively?"),
            DialogueTurn("t14", "I made the changes, and they approved and merged it! My code is officially in production. I am celebrating in my living room!"),
            DialogueTurn("t15", "I notice a lot of institutional knowledge isn't documented anywhere. Should I start writing documentation as a junior dev, or is that overstepping?"),
            DialogueTurn("t16", "Help me draft a polite message proposing a new 'Developer Wiki' page to my manager to show initiative."),
            DialogueTurn("t17", "My manager loved the Wiki idea and put me in charge of building it. The company is flying us all to a retreat next month to meet in person!"),
            DialogueTurn("t18", "I'm a little socially anxious. How do I transition from being a 'Slack persona' to a real human being at an intense three-day offsite?"),
            DialogueTurn("t19", "The offsite was incredible. We played board games and I actually bonded with the senior developers. I feel like a real part of the team now."),
            DialogueTurn("t20", "I'm back home. I want to propose adopting a new testing framework I learned about. How do I pitch a technical change when I'm still the junior?"),
            DialogueTurn("t21", "I wrote a formal design document for it. The principal engineer reviewed it and gave it the green light. I am leading the implementation!"),
            DialogueTurn("t22", "We successfully migrated the tests. The build pipeline is 20% faster. My manager specifically shouted me out in the all-hands meeting."),
            DialogueTurn("t23", "We just hired a new bootcamp grad. My manager asked me to be their 'onboarding buddy'. It feels crazy that I am the mentor now."),
            DialogueTurn("t24", "Give me a checklist of everything I wish I had known on my first day, so I can give this new hire the best possible start."),
        ),
    ),

    # ==========================================
    # SKELETON 30: SOURDOUGH BAKING JOURNEY
    # ==========================================

    # LENGTH: 8 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="sourdough_baking_journey_8_a",
        bucket="culinary_skills",
        title="Sourdough Baking Journey (8-Turn A)",
        system_preamble=(
            "You are a culinary instructor helping the same user learn a fun, low-stakes cooking skill over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I finally decided to jump on the bandwagon and learn how to bake sourdough bread. I just mixed flour and water for my starter today!"),
            DialogueTurn("t2", "It's day three and my starter smells slightly like nail polish remover. Did I kill it already?"),
            DialogueTurn("t3", "Okay, it's bubbly and ready! The recipe calls for 'folding' the dough instead of kneading it. Why do we do that?"),
            DialogueTurn("t4", "My kitchen is really cold today, about 65 degrees. Will that ruin the rising time?"),
            DialogueTurn("t5", "I totally forgot I'm supposed to go to a movie in an hour. Can I just put the dough in the fridge and deal with it tomorrow?"),
            DialogueTurn("t6", "I'm ready to bake. I don't have a Dutch oven, but I have a pizza stone and a regular baking sheet. Which is better?"),
            DialogueTurn("t7", "It just came out of the oven! It didn't rise quite as high as the pictures, but it sounds hollow when I tap it. Is it done?"),
            DialogueTurn("t8", "Give me a timeline for exactly how I should feed and store my leftover starter so I can bake again next weekend."),
        ),
    ),

    # LENGTH: 8 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="sourdough_baking_journey_8_b",
        bucket="culinary_skills",
        title="Sourdough Baking Journey (8-Turn B)",
        system_preamble=(
            "You are a culinary instructor helping the same user learn a fun, low-stakes cooking skill over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "A friend just gifted me a jar of their mature sourdough starter. I've never baked bread in my life. What is the absolute first thing I do with this?"),
            DialogueTurn("t2", "I fed it and it doubled in size. How do I do the 'float test' to see if it's ready to bake with?"),
            DialogueTurn("t3", "My recipe says to mix the flour and water and let it sit for an hour before adding the starter and salt. What does that do?"),
            DialogueTurn("t4", "I'm trying to shape the dough into a tight ball, but it's incredibly sticky and just coating my hands. Should I add more flour?"),
            DialogueTurn("t5", "I accidentally dropped my phone on the floor while my hands were covered in dough. How do I clean dried dough off a phone screen?"),
            DialogueTurn("t6", "The dough is in the proofing basket. When I flip it out tomorrow to bake, how deep should I score the top with a razor?"),
            DialogueTurn("t7", "I baked it! It looks gorgeous on top, but the very bottom crust is burnt black. What caused that?"),
            DialogueTurn("t8", "Draft a quick troubleshooting guide for me on how to prevent the bottom from burning on my next bake."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="sourdough_baking_journey_16_a",
        bucket="culinary_skills",
        title="Sourdough Baking Journey (16-Turn A)",
        system_preamble=(
            "You are a culinary instructor helping the same user learn a fun, low-stakes cooking skill over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I finally decided to jump on the bandwagon and learn how to bake sourdough bread. I just mixed flour and water for my starter today!"),
            DialogueTurn("t2", "It's day three and my starter smells slightly like nail polish remover. Did I kill it already?"),
            DialogueTurn("t3", "Okay, it's bubbly and ready! The recipe calls for 'folding' the dough instead of kneading it. Why do we do that?"),
            DialogueTurn("t4", "My kitchen is really cold today, about 65 degrees. Will that ruin the rising time?"),
            DialogueTurn("t5", "I totally forgot I'm supposed to go to a movie in an hour. Can I just put the dough in the fridge and deal with it tomorrow?"),
            DialogueTurn("t6", "I'm ready to bake. I don't have a Dutch oven, but I have a pizza stone and a regular baking sheet. Which is better?"),
            DialogueTurn("t7", "It just came out of the oven! It didn't rise quite as high as the pictures, but it sounds hollow when I tap it. Is it done?"),
            DialogueTurn("t8", "Give me a timeline for exactly how I should feed and store my leftover starter so I can bake again next weekend."),
            DialogueTurn("t9", "I'm accumulating so much 'discard' starter in my fridge. What are some easy recipes that use it so I don't waste flour?"),
            DialogueTurn("t10", "I made sourdough discard pancakes this morning and they were the best pancakes I've ever eaten!"),
            DialogueTurn("t11", "I want to try baking a loaf with 50% whole wheat flour instead of just white bread flour. Do I need to add more water?"),
            DialogueTurn("t12", "I added more water, but now the dough is a soupy mess. I've heard of 'slap and fold'. Will that help build tension in wet dough?"),
            DialogueTurn("t13", "The slap and fold method worked! The whole wheat loaf came out perfectly. The crumb is incredibly soft."),
            DialogueTurn("t14", "I want to bake a loaf to give to my neighbor as a thank you gift. Is there a trick to making the crust stay crusty in a plastic bag?"),
            DialogueTurn("t15", "I gifted the bread in a paper bag. My neighbor said it looked like it came from an artisanal bakery. I am so proud."),
            DialogueTurn("t16", "Help me draft a cute little label I can print out to put on future gifted loaves, including instructions on how they should store it."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="sourdough_baking_journey_16_b",
        bucket="culinary_skills",
        title="Sourdough Baking Journey (16-Turn B)",
        system_preamble=(
            "You are a culinary instructor helping the same user learn a fun, low-stakes cooking skill over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "A friend just gifted me a jar of their mature sourdough starter. I've never baked bread in my life. What is the absolute first thing I do with this?"),
            DialogueTurn("t2", "I fed it and it doubled in size. How do I do the 'float test' to see if it's ready to bake with?"),
            DialogueTurn("t3", "My recipe says to mix the flour and water and let it sit for an hour before adding the starter and salt. What does that do?"),
            DialogueTurn("t4", "I'm trying to shape the dough into a tight ball, but it's incredibly sticky and just coating my hands. Should I add more flour?"),
            DialogueTurn("t5", "I accidentally dropped my phone on the floor while my hands were covered in dough. How do I clean dried dough off a phone screen?"),
            DialogueTurn("t6", "The dough is in the proofing basket. When I flip it out tomorrow to bake, how deep should I score the top with a razor?"),
            DialogueTurn("t7", "I baked it! It looks gorgeous on top, but the very bottom crust is burnt black. What caused that?"),
            DialogueTurn("t8", "Draft a quick troubleshooting guide for me on how to prevent the bottom from burning on my next bake."),
            DialogueTurn("t9", "I bought a Dutch oven. My second plain loaf was flawless. Now I want to try adding inclusions. How do I make a jalapeño cheddar loaf?"),
            DialogueTurn("t10", "At what point in the folding process do I physically add the diced jalapeños and cheese so they don't tear the dough?"),
            DialogueTurn("t11", "I baked it! It smells amazing, but a lot of the cheese melted and burned on the outside of the crust. How do I keep it inside next time?"),
            DialogueTurn("t12", "The inside of the bread is perfect. I posted a picture on Instagram and three friends immediately asked if I sell them."),
            DialogueTurn("t13", "Should I scale up my recipe to bake two loaves at once? Does doubling the ingredients change the rising times?"),
            DialogueTurn("t14", "I baked two loaves simultaneously. It was stressful managing the timing, but pulling two perfect boules out of the oven is thrilling."),
            DialogueTurn("t15", "I gave the extra loaf away. Baking feels like such a tangible, generous hobby compared to my digital day job."),
            DialogueTurn("t16", "Give me a checklist of specialized tools (like a lame or a dough whisk) I should ask for for my birthday to upgrade my baking game."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="sourdough_baking_journey_24_a",
        bucket="culinary_skills",
        title="Sourdough Baking Journey (24-Turn A)",
        system_preamble=(
            "You are a culinary instructor helping the same user learn a fun, low-stakes cooking skill over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I finally decided to jump on the bandwagon and learn how to bake sourdough bread. I just mixed flour and water for my starter today!"),
            DialogueTurn("t2", "It's day three and my starter smells slightly like nail polish remover. Did I kill it already?"),
            DialogueTurn("t3", "Okay, it's bubbly and ready! The recipe calls for 'folding' the dough instead of kneading it. Why do we do that?"),
            DialogueTurn("t4", "My kitchen is really cold today, about 65 degrees. Will that ruin the rising time?"),
            DialogueTurn("t5", "I totally forgot I'm supposed to go to a movie in an hour. Can I just put the dough in the fridge and deal with it tomorrow?"),
            DialogueTurn("t6", "I'm ready to bake. I don't have a Dutch oven, but I have a pizza stone and a regular baking sheet. Which is better?"),
            DialogueTurn("t7", "It just came out of the oven! It didn't rise quite as high as the pictures, but it sounds hollow when I tap it. Is it done?"),
            DialogueTurn("t8", "Give me a timeline for exactly how I should feed and store my leftover starter so I can bake again next weekend."),
            DialogueTurn("t9", "I'm accumulating so much 'discard' starter in my fridge. What are some easy recipes that use it so I don't waste flour?"),
            DialogueTurn("t10", "I made sourdough discard pancakes this morning and they were the best pancakes I've ever eaten!"),
            DialogueTurn("t11", "I want to try baking a loaf with 50% whole wheat flour instead of just white bread flour. Do I need to add more water?"),
            DialogueTurn("t12", "I added more water, but now the dough is a soupy mess. I've heard of 'slap and fold'. Will that help build tension in wet dough?"),
            DialogueTurn("t13", "The slap and fold method worked! The whole wheat loaf came out perfectly. The crumb is incredibly soft."),
            DialogueTurn("t14", "I want to bake a loaf to give to my neighbor as a thank you gift. Is there a trick to making the crust stay crusty in a plastic bag?"),
            DialogueTurn("t15", "I gifted the bread in a paper bag. My neighbor said it looked like it came from an artisanal bakery. I am so proud."),
            DialogueTurn("t16", "Help me draft a cute little label I can print out to put on future gifted loaves, including instructions on how they should store it."),
            DialogueTurn("t17", "I left my starter on the counter for three weeks while I was traveling. It has a layer of dark liquid on top. Is it completely dead?"),
            DialogueTurn("t18", "I poured off the hooch and fed it for three days. It's bubbling again! Sourdough is incredibly resilient."),
            DialogueTurn("t19", "I'm ready to branch out. Can I use my starter to make sourdough pizza dough?"),
            DialogueTurn("t20", "The pizza dough is stretched. What temperature should my oven be at to get a really crispy crust on a home pizza stone?"),
            DialogueTurn("t21", "We had a pizza party. The crust was incredible—chewy and sour and perfectly charred. I'm never ordering delivery again."),
            DialogueTurn("t22", "I realize that the 24-hour process of making bread has actually become a form of moving meditation for me. It forces me to slow down."),
            DialogueTurn("t23", "I want to bake fresh bread for Thanksgiving dinner next week. It feels like a big responsibility, but I'm ready."),
            DialogueTurn("t24", "Give me a reverse-engineered timeline starting from 'warm bread on the table at 4 PM Thursday' so I know exactly when to start the levain on Tuesday."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="sourdough_baking_journey_24_b",
        bucket="culinary_skills",
        title="Sourdough Baking Journey (24-Turn B)",
        system_preamble=(
            "You are a culinary instructor helping the same user learn a fun, low-stakes cooking skill over several turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "A friend just gifted me a jar of their mature sourdough starter. I've never baked bread in my life. What is the absolute first thing I do with this?"),
            DialogueTurn("t2", "I fed it and it doubled in size. How do I do the 'float test' to see if it's ready to bake with?"),
            DialogueTurn("t3", "My recipe says to mix the flour and water and let it sit for an hour before adding the starter and salt. What does that do?"),
            DialogueTurn("t4", "I'm trying to shape the dough into a tight ball, but it's incredibly sticky and just coating my hands. Should I add more flour?"),
            DialogueTurn("t5", "I accidentally dropped my phone on the floor while my hands were covered in dough. How do I clean dried dough off a phone screen?"),
            DialogueTurn("t6", "The dough is in the proofing basket. When I flip it out tomorrow to bake, how deep should I score the top with a razor?"),
            DialogueTurn("t7", "I baked it! It looks gorgeous on top, but the very bottom crust is burnt black. What caused that?"),
            DialogueTurn("t8", "Draft a quick troubleshooting guide for me on how to prevent the bottom from burning on my next bake."),
            DialogueTurn("t9", "I bought a Dutch oven. My second plain loaf was flawless. Now I want to try adding inclusions. How do I make a jalapeño cheddar loaf?"),
            DialogueTurn("t10", "At what point in the folding process do I physically add the diced jalapeños and cheese so they don't tear the dough?"),
            DialogueTurn("t11", "I baked it! It smells amazing, but a lot of the cheese melted and burned on the outside of the crust. How do I keep it inside next time?"),
            DialogueTurn("t12", "The inside of the bread is perfect. I posted a picture on Instagram and three friends immediately asked if I sell them."),
            DialogueTurn("t13", "Should I scale up my recipe to bake two loaves at once? Does doubling the ingredients change the rising times?"),
            DialogueTurn("t14", "I baked two loaves simultaneously. It was stressful managing the timing, but pulling two perfect boules out of the oven is thrilling."),
            DialogueTurn("t15", "I gave the extra loaf away. Baking feels like such a tangible, generous hobby compared to my digital day job."),
            DialogueTurn("t16", "Give me a checklist of specialized tools (like a lame or a dough whisk) I should ask for for my birthday to upgrade my baking game."),
            DialogueTurn("t17", "My local farmers market said I could set up a small table to sell bread on weekends. What are 'cottage food laws' and do I need to worry about them?"),
            DialogueTurn("t18", "I checked the laws and I'm legally clear to sell. How do I calculate a fair price for a loaf of artisanal sourdough?"),
            DialogueTurn("t19", "I priced them at $10 each. I'm baking 10 loaves for the market tomorrow. My kitchen is completely covered in flour."),
            DialogueTurn("t20", "What is the most cost-effective and aesthetic way to package the loaves for sale?"),
            DialogueTurn("t21", "The market was a blur. I sold all 10 loaves in the first two hours! People actually paid money for something I made with my hands."),
            DialogueTurn("t22", "I'm exhausted. Baking 10 loaves in a home oven took me all night. I don't think I want to do this every weekend."),
            DialogueTurn("t23", "I've decided to keep it as a hobby. The joy was in giving it away, not the stress of production deadlines. I feel good about that choice."),
            DialogueTurn("t24", "Help me draft a polite, appreciative social media post announcing that I'm stepping back from selling, but still sharing my baking journey online."),
        ),
    ),

    # ==========================================
    # SKELETON 31: SURPRISE PARTY PLANNING
    # ==========================================

    # LENGTH: 8 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="surprise_party_planning_8_a",
        bucket="events",
        title="Surprise Party Planning (8-Turn A)",
        system_preamble=(
            "You are helping the same user plan a surprise party over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I want to plan a surprise 30th birthday for my spouse with our closest friends. How do I even start without them noticing?"),
            DialogueTurn("t2", "I think an intimate dinner party at our favorite restaurant is best. How do I secretly get their friends' contact info?"),
            DialogueTurn("t3", "I got the guest list together! Everyone is so excited. What's a good cover story for why we are going out that night?"),
            DialogueTurn("t4", "I told them my boss gave us a gift certificate. Now, should I handle the menu ahead of time or let people order?"),
            DialogueTurn("t5", "The restaurant wants a deposit. How do I pay for this without it showing up on our joint credit card statement?"),
            DialogueTurn("t6", "I want to do something sentimental, like a slideshow. Is it too much to ask the guests to send me photos in advance?"),
            DialogueTurn("t7", "I've got a great mix of funny and sweet photos. What's a good song to play in the background?"),
            DialogueTurn("t8", "Give me a checklist of three things I need to double-check with the restaurant the week of the party."),
        ),
    ),

    # LENGTH: 8 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="surprise_party_planning_8_b",
        bucket="events",
        title="Surprise Party Planning (8-Turn B)",
        system_preamble=(
            "You are helping the same user plan a surprise party over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I've been tasked with planning a surprise retirement party for a coworker who has been here 20 years. What are the logistics I need to tackle first?"),
            DialogueTurn("t2", "We have a budget of $500 from the company. How do I stretch that to feed about 40 people in the office breakroom?"),
            DialogueTurn("t3", "I like the idea of a taco bar. How do I make sure people don't schedule over the meeting block on their calendars?"),
            DialogueTurn("t4", "We sent a fake 'Mandatory All-Hands' invite. Now, how do we physically decorate the room without the retiree walking in?"),
            DialogueTurn("t5", "I need to order a cake. Should I get a generic one or something specific to their hobbies? They like golfing."),
            DialogueTurn("t6", "We want to get a group gift. What is the most polite way to ask the office for cash contributions without being pushy?"),
            DialogueTurn("t7", "Someone from another department wants to give a 15-minute speech. How do I politely tell them we need to keep it to 2 minutes?"),
            DialogueTurn("t8", "Draft a quick run-of-show schedule for the one-hour party block."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="surprise_party_planning_16_a",
        bucket="events",
        title="Surprise Party Planning (16-Turn A)",
        system_preamble=(
            "You are helping the same user plan a surprise party over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I want to plan a surprise 30th birthday for my spouse with our closest friends. How do I even start without them noticing?"),
            DialogueTurn("t2", "I think an intimate dinner party at our favorite restaurant is best. How do I secretly get their friends' contact info?"),
            DialogueTurn("t3", "I got the guest list together! Everyone is so excited. What's a good cover story for why we are going out that night?"),
            DialogueTurn("t4", "I told them my boss gave us a gift certificate. Now, should I handle the menu ahead of time or let people order?"),
            DialogueTurn("t5", "The restaurant wants a deposit. How do I pay for this without it showing up on our joint credit card statement?"),
            DialogueTurn("t6", "I want to do something sentimental, like a slideshow. Is it too much to ask the guests to send me photos in advance?"),
            DialogueTurn("t7", "I've got a great mix of funny and sweet photos. What's a good song to play in the background?"),
            DialogueTurn("t8", "Give me a checklist of three things I need to double-check with the restaurant the week of the party."),
            DialogueTurn("t9", "Disaster. The restaurant just called and said they had a kitchen fire. They have to cancel our reservation. The party is in 5 days!"),
            DialogueTurn("t10", "I am completely panicking. I don't think I can find a private room on this short notice. Should I just cancel the whole thing?"),
            DialogueTurn("t11", "Okay, deep breaths. A friend offered to host it at their house instead. How do I quickly pivot the catering plan without spending a fortune?"),
            DialogueTurn("t12", "I ordered drop-off catering from a local Italian place. But now, how do I get my spouse to our friend's house instead of the restaurant without being suspicious?"),
            DialogueTurn("t13", "The cover story worked. But now my spouse is trying to make plans with the exact friends who are supposed to be at the party. How do I tell the friends to act normal?"),
            DialogueTurn("t14", "My anxiety is through the roof. I am so scared the secret is going to slip. Tell me this is going to be worth it."),
            DialogueTurn("t15", "It's the day of the party. The food is delivered, everyone is hiding in the living room. We are in the driveway. What do I do right before we walk in?"),
            DialogueTurn("t16", "We did it! They were completely shocked and cried happy tears. The pivot to the house actually made it cozier. Help me draft a quick thank-you text to the friend who hosted."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="surprise_party_planning_16_b",
        bucket="events",
        title="Surprise Party Planning (16-Turn B)",
        system_preamble=(
            "You are helping the same user plan a surprise party over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I've been tasked with planning a surprise retirement party for a coworker who has been here 20 years. What are the logistics I need to tackle first?"),
            DialogueTurn("t2", "We have a budget of $500 from the company. How do I stretch that to feed about 40 people in the office breakroom?"),
            DialogueTurn("t3", "I like the idea of a taco bar. How do I make sure people don't schedule over the meeting block on their calendars?"),
            DialogueTurn("t4", "We sent a fake 'Mandatory All-Hands' invite. Now, how do we physically decorate the room without the retiree walking in?"),
            DialogueTurn("t5", "I need to order a cake. Should I get a generic one or something specific to their hobbies? They like golfing."),
            DialogueTurn("t6", "We want to get a group gift. What is the most polite way to ask the office for cash contributions without being pushy?"),
            DialogueTurn("t7", "Someone from another department wants to give a 15-minute speech. How do I politely tell them we need to keep it to 2 minutes?"),
            DialogueTurn("t8", "Draft a quick run-of-show schedule for the one-hour party block."),
            DialogueTurn("t9", "We hit a snag. HR just reminded me of a strict new policy about serving alcohol on company property. We can't do the champagne toast. What's a festive alternative?"),
            DialogueTurn("t10", "Sparkling cider it is. But now three people on my team suddenly remembered they have severe gluten and dairy allergies. The taco bar is already ordered. What do I do?"),
            DialogueTurn("t11", "I managed to add some allergy-friendly sides. Honestly, wrangling these coworkers is like herding cats. Half of them haven't RSVP'd to the fake meeting."),
            DialogueTurn("t12", "I'll send a direct ping to the stragglers. Wait, the retiree just told our boss they might take the day off on the day of the party! How do we stop them?"),
            DialogueTurn("t13", "The boss convinced them they are needed for a 'critical client handoff'. Crisis averted. But now I have to coordinate getting the gift delivered to the office discreetly."),
            DialogueTurn("t14", "It's party day. I am stressed, sweaty, and hiding balloons under my desk. The caterer is running 10 minutes late. Do I delay the fake meeting?"),
            DialogueTurn("t15", "The food arrived just in time. The retiree is walking down the hall right now. Give me a one-sentence cue to yell to the room."),
            DialogueTurn("t16", "The party was a hit. The retiree was genuinely surprised and loved the golf gift. I'm exhausted. Draft a quick email to the whole staff thanking them for keeping the secret."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="surprise_party_planning_24_a",
        bucket="events",
        title="Surprise Party Planning (24-Turn A)",
        system_preamble=(
            "You are helping the same user plan a surprise party over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I want to plan a surprise 30th birthday for my spouse with our closest friends. How do I even start without them noticing?"),
            DialogueTurn("t2", "I think an intimate dinner party at our favorite restaurant is best. How do I secretly get their friends' contact info?"),
            DialogueTurn("t3", "I got the guest list together! Everyone is so excited. What's a good cover story for why we are going out that night?"),
            DialogueTurn("t4", "I told them my boss gave us a gift certificate. Now, should I handle the menu ahead of time or let people order?"),
            DialogueTurn("t5", "The restaurant wants a deposit. How do I pay for this without it showing up on our joint credit card statement?"),
            DialogueTurn("t6", "I want to do something sentimental, like a slideshow. Is it too much to ask the guests to send me photos in advance?"),
            DialogueTurn("t7", "I've got a great mix of funny and sweet photos. What's a good song to play in the background?"),
            DialogueTurn("t8", "Give me a checklist of three things I need to double-check with the restaurant the week of the party."),
            DialogueTurn("t9", "Disaster. The restaurant just called and said they had a kitchen fire. They have to cancel our reservation. The party is in 5 days!"),
            DialogueTurn("t10", "I am completely panicking. I don't think I can find a private room on this short notice. Should I just cancel the whole thing?"),
            DialogueTurn("t11", "Okay, deep breaths. A friend offered to host it at their house instead. How do I quickly pivot the catering plan without spending a fortune?"),
            DialogueTurn("t12", "I ordered drop-off catering from a local Italian place. But now, how do I get my spouse to our friend's house instead of the restaurant without being suspicious?"),
            DialogueTurn("t13", "The cover story worked. But now my spouse is trying to make plans with the exact friends who are supposed to be at the party. How do I tell the friends to act normal?"),
            DialogueTurn("t14", "My anxiety is through the roof. I am so scared the secret is going to slip. Tell me this is going to be worth it."),
            DialogueTurn("t15", "It's the day of the party. The food is delivered, everyone is hiding in the living room. We are in the driveway. What do I do right before we walk in?"),
            DialogueTurn("t16", "We did it! They were completely shocked and cried happy tears. The pivot to the house actually made it cozier. Help me draft a quick thank-you text to the friend who hosted."),
            DialogueTurn("t17", "We are looking through all the photos from last night. It was so perfect. I feel a huge wave of relief. How do I politely share the photos with the guests?"),
            DialogueTurn("t18", "One of the guests left a really nice jacket at the house. What's the polite etiquette for getting it back to them?"),
            DialogueTurn("t19", "My spouse is still on cloud nine. They want to plan a 'thank you' brunch for everyone who helped. Is that overkill?"),
            DialogueTurn("t20", "We decided to just do a casual Sunday coffee at our place instead. What are some easy pastries I can bake from scratch to show effort?"),
            DialogueTurn("t21", "The coffee morning was lovely. It was great to actually talk to everyone without the stress of keeping a secret. I feel really proud of myself."),
            DialogueTurn("t22", "I think I want to make a small physical photo album of the night to keep on our coffee table. What's a good service for printing high-quality photobooks?"),
            DialogueTurn("t23", "The book arrived! It looks incredibly professional. I almost can't believe I pulled this whole thing off without a hitch."),
            DialogueTurn("t24", "My spouse's 31st is next year and they jokingly said 'good luck topping this.' Give me a funny, extremely low-effort idea for next year to reset expectations."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="surprise_party_planning_24_b",
        bucket="events",
        title="Surprise Party Planning (24-Turn B)",
        system_preamble=(
            "You are helping the same user plan a surprise party over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I've been tasked with planning a surprise retirement party for a coworker who has been here 20 years. What are the logistics I need to tackle first?"),
            DialogueTurn("t2", "We have a budget of $500 from the company. How do I stretch that to feed about 40 people in the office breakroom?"),
            DialogueTurn("t3", "I like the idea of a taco bar. How do I make sure people don't schedule over the meeting block on their calendars?"),
            DialogueTurn("t4", "We sent a fake 'Mandatory All-Hands' invite. Now, how do we physically decorate the room without the retiree walking in?"),
            DialogueTurn("t5", "I need to order a cake. Should I get a generic one or something specific to their hobbies? They like golfing."),
            DialogueTurn("t6", "We want to get a group gift. What is the most polite way to ask the office for cash contributions without being pushy?"),
            DialogueTurn("t7", "Someone from another department wants to give a 15-minute speech. How do I politely tell them we need to keep it to 2 minutes?"),
            DialogueTurn("t8", "Draft a quick run-of-show schedule for the one-hour party block."),
            DialogueTurn("t9", "We hit a snag. HR just reminded me of a strict new policy about serving alcohol on company property. We can't do the champagne toast. What's a festive alternative?"),
            DialogueTurn("t10", "Sparkling cider it is. But now three people on my team suddenly remembered they have severe gluten and dairy allergies. The taco bar is already ordered. What do I do?"),
            DialogueTurn("t11", "I managed to add some allergy-friendly sides. Honestly, wrangling these coworkers is like herding cats. Half of them haven't RSVP'd to the fake meeting."),
            DialogueTurn("t12", "I'll send a direct ping to the stragglers. Wait, the retiree just told our boss they might take the day off on the day of the party! How do we stop them?"),
            DialogueTurn("t13", "The boss convinced them they are needed for a 'critical client handoff'. Crisis averted. But now I have to coordinate getting the gift delivered to the office discreetly."),
            DialogueTurn("t14", "It's party day. I am stressed, sweaty, and hiding balloons under my desk. The caterer is running 10 minutes late. Do I delay the fake meeting?"),
            DialogueTurn("t15", "The food arrived just in time. The retiree is walking down the hall right now. Give me a one-sentence cue to yell to the room."),
            DialogueTurn("t16", "The party was a hit. The retiree was genuinely surprised and loved the golf gift. I'm exhausted. Draft a quick email to the whole staff thanking them for keeping the secret."),
            DialogueTurn("t17", "Now I have to process the expense reports. Our finance team is notoriously strict. What do I do about a missing $12 receipt for napkins?"),
            DialogueTurn("t18", "I got the expenses approved. But now the retiree's desk is completely empty and it feels weirdly depressing in the office. How do we boost morale?"),
            DialogueTurn("t19", "We are going to do an informal team lunch. Speaking of the empty desk, they put me in charge of archiving their old paper files. It's a mountain. Where do I begin?"),
            DialogueTurn("t20", "I found an incredibly outdated company manual from 1998 in their files. It's hilarious. Should I scan it and send it to the team for a laugh?"),
            DialogueTurn("t21", "The team loved the 1998 manual. It really broke the tension. The retiree actually sent me a personal thank-you card in the mail. It was incredibly sweet."),
            DialogueTurn("t22", "I feel like I've gained a reputation as the 'office party planner' now. Someone just asked me to organize a baby shower. How do I politely decline?"),
            DialogueTurn("t23", "I successfully set a boundary! I passed the baby shower off to someone else. It feels good to just do my actual job today."),
            DialogueTurn("t24", "Write a short, professional bulleted list of 'Lessons Learned in Event Planning' that I can casually drop in the slack channel for the person planning the baby shower."),
        ),
    ),

    # ==========================================
    # SKELETON 32: SCI-FI STORY BRAINSTORM
    # ==========================================

    # LENGTH: 8 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="sci_fi_story_brainstorm_8_a",
        bucket="creative",
        title="Sci-Fi Story Brainstorm (8-Turn A)",
        system_preamble=(
            "You are helping the same user brainstorm a science fiction story over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I want to write a hard sci-fi story about a lone astronaut stranded in the asteroid belt, but I want the physics to be realistic. Where do we start?"),
            DialogueTurn("t2", "Let's figure out the inciting incident. How could a mining ship suffer a catastrophic failure without immediately killing the pilot?"),
            DialogueTurn("t3", "A micro-meteorite strike taking out the main comms and navigation sounds perfect. How much oxygen and water would they realistically have left?"),
            DialogueTurn("t4", "So they have three weeks of air. What is a realistic, scientifically plausible way they could try to signal for help using mining equipment?"),
            DialogueTurn("t5", "Modifying the mining laser to pulse a distress signal is a great idea. But I need a complication. What goes wrong when they try to fix it?"),
            DialogueTurn("t6", "Okay, they expose the power core and it causes a slow leak in the ship's heating system. How cold does it get in the asteroid belt?"),
            DialogueTurn("t7", "This is getting tense. They have to cannibalize their spacesuit's thermal lining to patch the leak. What's the emotional state of the character right now?"),
            DialogueTurn("t8", "Help me draft the opening paragraph describing the terrifying silence right after the meteor strike hits."),
        ),
    ),

    # LENGTH: 8 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="sci_fi_story_brainstorm_8_b",
        bucket="creative",
        title="Sci-Fi Story Brainstorm (8-Turn B)",
        system_preamble=(
            "You are helping the same user brainstorm a science fiction story over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I want to write a sweeping space opera. Think galactic empires, royal families, and rebellion. What is a good core conflict to build the universe around?"),
            DialogueTurn("t2", "A monopoly on a faster-than-light travel resource is great. Let's make the protagonist a runaway heir to the empire. Why did they flee?"),
            DialogueTurn("t3", "They discovered the FTL resource is actually destroying the fabric of space. They run into a ragtag group of smugglers. Who is the captain of this ship?"),
            DialogueTurn("t4", "A cynical ex-military pilot who doesn't trust royals. I love it. How do we build romantic tension between these two completely opposite characters?"),
            DialogueTurn("t5", "Forced proximity during a space battle! What kind of alien faction attacks them, and what do their ships look like?"),
            DialogueTurn("t6", "Sleek, obsidian ships that use gravity weapons. The smugglers get captured. What does the interior of the alien prison look like?"),
            DialogueTurn("t7", "I want the royal heir to reveal their identity to save the smuggler captain. How can we make this moment dramatically impactful?"),
            DialogueTurn("t8", "Draft the dialogue exchange where the captain realizes the person they've been arguing with is actually the heir to the throne."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="sci_fi_story_brainstorm_16_a",
        bucket="creative",
        title="Sci-Fi Story Brainstorm (16-Turn A)",
        system_preamble=(
            "You are helping the same user brainstorm a science fiction story over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I want to write a hard sci-fi story about a lone astronaut stranded in the asteroid belt, but I want the physics to be realistic. Where do we start?"),
            DialogueTurn("t2", "Let's figure out the inciting incident. How could a mining ship suffer a catastrophic failure without immediately killing the pilot?"),
            DialogueTurn("t3", "A micro-meteorite strike taking out the main comms and navigation sounds perfect. How much oxygen and water would they realistically have left?"),
            DialogueTurn("t4", "So they have three weeks of air. What is a realistic, scientifically plausible way they could try to signal for help using mining equipment?"),
            DialogueTurn("t5", "Modifying the mining laser to pulse a distress signal is a great idea. But I need a complication. What goes wrong when they try to fix it?"),
            DialogueTurn("t6", "Okay, they expose the power core and it causes a slow leak in the ship's heating system. How cold does it get in the asteroid belt?"),
            DialogueTurn("t7", "This is getting tense. They have to cannibalize their spacesuit's thermal lining to patch the leak. What's the emotional state of the character right now?"),
            DialogueTurn("t8", "Help me draft the opening paragraph describing the terrifying silence right after the meteor strike hits."),
            DialogueTurn("t9", "I've hit a wall. I realized that if the navigation is dead, pulsing a laser blindly into space is mathematically useless. The odds of hitting a receiver are zero. The plot is broken."),
            DialogueTurn("t10", "I'm so frustrated. I feel like abandoning this draft. I can't write 'hard' sci-fi if the central premise of survival is physically impossible. What do I do?"),
            DialogueTurn("t11", "Wait. What if they use the laser to hit a specific, known relay station on Ceres? How much power would that take compared to a blind pulse?"),
            DialogueTurn("t12", "Okay, they need to manually calculate the orbital trajectory of Ceres using just a star chart and a stopwatch. Is that actually possible?"),
            DialogueTurn("t13", "This is brilliant! It turns the plot hole into a massive test of their skill. The tension of waiting to see if the math was right is perfect."),
            DialogueTurn("t14", "I'm writing the scene where they finally fire the laser. I feel so inspired again. I'm literally typing so fast my hands hurt."),
            DialogueTurn("t15", "The signal is received. A rescue drone is dispatched, but they are down to 1% oxygen. How do they survive the final 12 hours?"),
            DialogueTurn("t16", "They did it. They survived by putting themselves into a hypothermic state. Draft the final, closing sentence of the story as they see the rescue lights."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="sci_fi_story_brainstorm_16_b",
        bucket="creative",
        title="Sci-Fi Story Brainstorm (16-Turn B)",
        system_preamble=(
            "You are helping the same user brainstorm a science fiction story over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I want to write a sweeping space opera. Think galactic empires, royal families, and rebellion. What is a good core conflict to build the universe around?"),
            DialogueTurn("t2", "A monopoly on a faster-than-light travel resource is great. Let's make the protagonist a runaway heir to the empire. Why did they flee?"),
            DialogueTurn("t3", "They discovered the FTL resource is actually destroying the fabric of space. They run into a ragtag group of smugglers. Who is the captain of this ship?"),
            DialogueTurn("t4", "A cynical ex-military pilot who doesn't trust royals. I love it. How do we build romantic tension between these two completely opposite characters?"),
            DialogueTurn("t5", "Forced proximity during a space battle! What kind of alien faction attacks them, and what do their ships look like?"),
            DialogueTurn("t6", "Sleek, obsidian ships that use gravity weapons. The smugglers get captured. What does the interior of the alien prison look like?"),
            DialogueTurn("t7", "I want the royal heir to reveal their identity to save the smuggler captain. How can we make this moment dramatically impactful?"),
            DialogueTurn("t8", "Draft the dialogue exchange where the captain realizes the person they've been arguing with is actually the heir to the throne."),
            DialogueTurn("t9", "I'm stuck. The romantic subplot is completely taking over the story. The pacing feels like a soap opera now, not an epic space adventure. I hate it."),
            DialogueTurn("t10", "I feel like I need to kill a main character to inject some actual stakes back into the plot. Should I kill the captain? I'm honestly so annoyed with my own writing right now."),
            DialogueTurn("t11", "You're right, killing the captain is a cheap trick. What if they get separated instead? One stuck on a prison planet, the other leading a fleet. How do we split them up logically?"),
            DialogueTurn("t12", "An exploding jump-gate! Yes! Now we have two parallel storylines. This gives the universe so much more scale. I'm feeling better about this."),
            DialogueTurn("t13", "The heir has to rally the outer rim worlds. What is a compelling argument they can use to unite factions that have hated each other for centuries?"),
            DialogueTurn("t14", "I just finished outlining the third act. The two storylines converge in a massive battle over the imperial capital. I am incredibly pumped to write this."),
            DialogueTurn("t15", "The captain sacrifices their ship to take down the planetary shield. It's so tragic but beautiful. How does the heir honor them in the aftermath?"),
            DialogueTurn("t16", "The story is finished. 120,000 words. I am crying. Please draft a short, poignant dedication page for the front of the book."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="sci_fi_story_brainstorm_24_a",
        bucket="creative",
        title="Sci-Fi Story Brainstorm (24-Turn A)",
        system_preamble=(
            "You are helping the same user brainstorm a science fiction story over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I want to write a hard sci-fi story about a lone astronaut stranded in the asteroid belt, but I want the physics to be realistic. Where do we start?"),
            DialogueTurn("t2", "Let's figure out the inciting incident. How could a mining ship suffer a catastrophic failure without immediately killing the pilot?"),
            DialogueTurn("t3", "A micro-meteorite strike taking out the main comms and navigation sounds perfect. How much oxygen and water would they realistically have left?"),
            DialogueTurn("t4", "So they have three weeks of air. What is a realistic, scientifically plausible way they could try to signal for help using mining equipment?"),
            DialogueTurn("t5", "Modifying the mining laser to pulse a distress signal is a great idea. But I need a complication. What goes wrong when they try to fix it?"),
            DialogueTurn("t6", "Okay, they expose the power core and it causes a slow leak in the ship's heating system. How cold does it get in the asteroid belt?"),
            DialogueTurn("t7", "This is getting tense. They have to cannibalize their spacesuit's thermal lining to patch the leak. What's the emotional state of the character right now?"),
            DialogueTurn("t8", "Help me draft the opening paragraph describing the terrifying silence right after the meteor strike hits."),
            DialogueTurn("t9", "I've hit a wall. I realized that if the navigation is dead, pulsing a laser blindly into space is mathematically useless. The odds of hitting a receiver are zero. The plot is broken."),
            DialogueTurn("t10", "I'm so frustrated. I feel like abandoning this draft. I can't write 'hard' sci-fi if the central premise of survival is physically impossible. What do I do?"),
            DialogueTurn("t11", "Wait. What if they use the laser to hit a specific, known relay station on Ceres? How much power would that take compared to a blind pulse?"),
            DialogueTurn("t12", "Okay, they need to manually calculate the orbital trajectory of Ceres using just a star chart and a stopwatch. Is that actually possible?"),
            DialogueTurn("t13", "This is brilliant! It turns the plot hole into a massive test of their skill. The tension of waiting to see if the math was right is perfect."),
            DialogueTurn("t14", "I'm writing the scene where they finally fire the laser. I feel so inspired again. I'm literally typing so fast my hands hurt."),
            DialogueTurn("t15", "The signal is received. A rescue drone is dispatched, but they are down to 1% oxygen. How do they survive the final 12 hours?"),
            DialogueTurn("t16", "They did it. They survived by putting themselves into a hypothermic state. Draft the final, closing sentence of the story as they see the rescue lights."),
            DialogueTurn("t17", "I finished the manuscript and sent it to a beta reader. They loved the science but said the protagonist feels a bit flat emotionally. How do I fix that in revisions?"),
            DialogueTurn("t18", "Adding flashbacks to their life on Earth before they became a miner is a great idea. What is a subtle regret they could be dwelling on while freezing?"),
            DialogueTurn("t19", "A strained relationship with an estranged sibling. That adds so much depth. I'm rewriting the mid-point right now to include this."),
            DialogueTurn("t20", "The revisions are done. The story feels so much richer now. I'm actually thinking about submitting this to a sci-fi magazine. How do I write a query letter?"),
            DialogueTurn("t21", "I sent out three queries today. I am incredibly nervous but also so proud of myself. I never thought I'd actually finish a story this long."),
            DialogueTurn("t22", "I got a rejection letter. It was a form rejection. It stings a little, but I'm trying not to let it ruin my week."),
            DialogueTurn("t23", "Wait! The second magazine just emailed. They want to buy the story! I am literally jumping around my living room. I'm a published author!"),
            DialogueTurn("t24", "Give me three one-sentence ideas for a completely different sci-fi short story so I can ride this momentum into a new project."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="sci_fi_story_brainstorm_24_b",
        bucket="creative",
        title="Sci-Fi Story Brainstorm (24-Turn B)",
        system_preamble=(
            "You are helping the same user brainstorm a science fiction story over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I want to write a sweeping space opera. Think galactic empires, royal families, and rebellion. What is a good core conflict to build the universe around?"),
            DialogueTurn("t2", "A monopoly on a faster-than-light travel resource is great. Let's make the protagonist a runaway heir to the empire. Why did they flee?"),
            DialogueTurn("t3", "They discovered the FTL resource is actually destroying the fabric of space. They run into a ragtag group of smugglers. Who is the captain of this ship?"),
            DialogueTurn("t4", "A cynical ex-military pilot who doesn't trust royals. I love it. How do we build romantic tension between these two completely opposite characters?"),
            DialogueTurn("t5", "Forced proximity during a space battle! What kind of alien faction attacks them, and what do their ships look like?"),
            DialogueTurn("t6", "Sleek, obsidian ships that use gravity weapons. The smugglers get captured. What does the interior of the alien prison look like?"),
            DialogueTurn("t7", "I want the royal heir to reveal their identity to save the smuggler captain. How can we make this moment dramatically impactful?"),
            DialogueTurn("t8", "Draft the dialogue exchange where the captain realizes the person they've been arguing with is actually the heir to the throne."),
            DialogueTurn("t9", "I'm stuck. The romantic subplot is completely taking over the story. The pacing feels like a soap opera now, not an epic space adventure. I hate it."),
            DialogueTurn("t10", "I feel like I need to kill a main character to inject some actual stakes back into the plot. Should I kill the captain? I'm honestly so annoyed with my own writing right now."),
            DialogueTurn("t11", "You're right, killing the captain is a cheap trick. What if they get separated instead? One stuck on a prison planet, the other leading a fleet. How do we split them up logically?"),
            DialogueTurn("t12", "An exploding jump-gate! Yes! Now we have two parallel storylines. This gives the universe so much more scale. I'm feeling better about this."),
            DialogueTurn("t13", "The heir has to rally the outer rim worlds. What is a compelling argument they can use to unite factions that have hated each other for centuries?"),
            DialogueTurn("t14", "I just finished outlining the third act. The two storylines converge in a massive battle over the imperial capital. I am incredibly pumped to write this."),
            DialogueTurn("t15", "The captain sacrifices their ship to take down the planetary shield. It's so tragic but beautiful. How does the heir honor them in the aftermath?"),
            DialogueTurn("t16", "The story is finished. 120,000 words. I am crying. Please draft a short, poignant dedication page for the front of the book."),
            DialogueTurn("t17", "I started formatting the manuscript for self-publishing on Amazon, but the formatting software is driving me insane. Everything looks broken."),
            DialogueTurn("t18", "Okay, I stepped away, watched a tutorial, and fixed the formatting. Now I need a cover. What are the key visual elements of a good space opera cover?"),
            DialogueTurn("t19", "I commissioned an artist for the cover and it looks stunning. Bright neon colors and massive ships. I'm setting up the launch page. What should the blurb say?"),
            DialogueTurn("t20", "The blurb is perfect. The book goes live in exactly 24 hours. I'm terrified no one is going to read it."),
            DialogueTurn("t21", "It's live! I sold ten copies on the first day, mostly to friends, but it's a start. How do I get genuine reviews from strangers?"),
            DialogueTurn("t22", "A book blogger on Twitter just reviewed it and gave it 4 stars! They said the worldbuilding was incredible. I'm absolutely ecstatic."),
            DialogueTurn("t23", "Sales are picking up. I have fans asking when the sequel is coming out. This is the craziest feeling in the world."),
            DialogueTurn("t24", "Give me a high-level summary of what the political landscape of the galaxy looks like five years after the end of book one, so I can start plotting the sequel."),
        ),
    ),

    # ==========================================
    # SKELETON 33: CASUAL 5K TRAINING
    # ==========================================

    # LENGTH: 8 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="casual_5k_training_8_a",
        bucket="habits",
        title="Casual 5K Training (8-Turn A)",
        system_preamble=(
            "You are helping the same user train for a 5K run over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I want to train for a 5K that's two months away. I'm a total beginner but I love data. What apps or gear should I start with?"),
            DialogueTurn("t2", "I downloaded Strava and got a heart rate monitor. I did my first run today. My heart rate spiked to 180 almost immediately. Is that bad?"),
            DialogueTurn("t3", "Okay, I need to run much slower. What is 'Zone 2' training and how do I calculate my zones?"),
            DialogueTurn("t4", "I tried staying in Zone 2 today, but I had to walk almost the whole time to keep my heart rate down. Is this normal?"),
            DialogueTurn("t5", "I've been sticking to the slow intervals for two weeks. My resting heart rate is already dropping. This data is so motivating!"),
            DialogueTurn("t6", "I want to try a 'speed day' just to test my limits. What is a simple interval workout I can do on a local track?"),
            DialogueTurn("t7", "The 400m intervals felt amazing. I felt like I was flying. How do I balance speed days with my slow days for the next month?"),
            DialogueTurn("t8", "Draft a sample training week for me that includes one speed day, two slow days, and a long run."),
        ),
    ),

    # LENGTH: 8 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="casual_5k_training_8_b",
        bucket="habits",
        title="Casual 5K Training (8-Turn B)",
        system_preamble=(
            "You are helping the same user train for a 5K run over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I want to run a 5K in a few months, but purely for my mental health. I want to build a routine without obsessing over times or data. Where do I start?"),
            DialogueTurn("t2", "I love the idea of running without a watch. I went for my first jog this morning. It was hard, but the fresh air was incredible."),
            DialogueTurn("t3", "I'm struggling to get out of bed when my alarm goes off. What's a low-pressure way to convince myself to just get out the door?"),
            DialogueTurn("t4", "Laying out my clothes the night before helped. Today I listened to an audiobook instead of music. It made the run feel like an escape."),
            DialogueTurn("t5", "My legs are feeling heavy today. Is it okay to just walk the whole route, or does that defeat the purpose of training?"),
            DialogueTurn("t6", "I walked the route and actually enjoyed noticing the trees and the neighborhood. I'm starting to look forward to these mornings."),
            DialogueTurn("t7", "I managed to run for 15 minutes straight today without stopping. I wasn't even trying to push it, my body just felt good."),
            DialogueTurn("t8", "Draft a brief, encouraging morning mantra I can say to myself when I lace up my shoes."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="casual_5k_training_16_a",
        bucket="habits",
        title="Casual 5K Training (16-Turn A)",
        system_preamble=(
            "You are helping the same user train for a 5K run over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I want to train for a 5K that's two months away. I'm a total beginner but I love data. What apps or gear should I start with?"),
            DialogueTurn("t2", "I downloaded Strava and got a heart rate monitor. I did my first run today. My heart rate spiked to 180 almost immediately. Is that bad?"),
            DialogueTurn("t3", "Okay, I need to run much slower. What is 'Zone 2' training and how do I calculate my zones?"),
            DialogueTurn("t4", "I tried staying in Zone 2 today, but I had to walk almost the whole time to keep my heart rate down. Is this normal?"),
            DialogueTurn("t5", "I've been sticking to the slow intervals for two weeks. My resting heart rate is already dropping. This data is so motivating!"),
            DialogueTurn("t6", "I want to try a 'speed day' just to test my limits. What is a simple interval workout I can do on a local track?"),
            DialogueTurn("t7", "The 400m intervals felt amazing. I felt like I was flying. How do I balance speed days with my slow days for the next month?"),
            DialogueTurn("t8", "Draft a sample training week for me that includes one speed day, two slow days, and a long run."),
            DialogueTurn("t9", "It's week four and the honeymoon phase is over. It's been raining all week, I got shin splints, and my times are actually getting worse. I feel defeated."),
            DialogueTurn("t10", "I haven't run in four days. I feel incredibly guilty and I'm losing all my motivation. What's the point if I'm just getting slower?"),
            DialogueTurn("t11", "You're right, rest is important for healing. What are some stretches I can do specifically for shin splints while I recover?"),
            DialogueTurn("t12", "My shins feel better. I want to try a very light, easy run today. How do I mentally accept that my pace is going to be slow after a week off?"),
            DialogueTurn("t13", "I finished the run. It was slow, but I did it. I feel a quiet sense of determination creeping back in. I'm not going to quit."),
            DialogueTurn("t14", "The race is next week. I just did a test run and broke my personal record! I am so hyped. How should I approach training in this final week?"),
            DialogueTurn("t15", "It's race morning. I am pinning my bib on right now. My stomach is full of butterflies. Any last-minute strategy tips for the start line?"),
            DialogueTurn("t16", "I FINISHED! Not only did I finish, I beat my goal time by two whole minutes! I am so incredibly proud of myself. Thank you for the guidance!"),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="casual_5k_training_16_b",
        bucket="habits",
        title="Casual 5K Training (16-Turn B)",
        system_preamble=(
            "You are helping the same user train for a 5K run over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I want to run a 5K in a few months, but purely for my mental health. I want to build a routine without obsessing over times or data. Where do I start?"),
            DialogueTurn("t2", "I love the idea of running without a watch. I went for my first jog this morning. It was hard, but the fresh air was incredible."),
            DialogueTurn("t3", "I'm struggling to get out of bed when my alarm goes off. What's a low-pressure way to convince myself to just get out the door?"),
            DialogueTurn("t4", "Laying out my clothes the night before helped. Today I listened to an audiobook instead of music. It made the run feel like an escape."),
            DialogueTurn("t5", "My legs are feeling heavy today. Is it okay to just walk the whole route, or does that defeat the purpose of training?"),
            DialogueTurn("t6", "I walked the route and actually enjoyed noticing the trees and the neighborhood. I'm starting to look forward to these mornings."),
            DialogueTurn("t7", "I managed to run for 15 minutes straight today without stopping. I wasn't even trying to push it, my body just felt good."),
            DialogueTurn("t8", "Draft a brief, encouraging morning mantra I can say to myself when I lace up my shoes."),
            DialogueTurn("t9", "Work has been incredibly stressful this week. I've skipped three runs in a row and I'm starting to spiral into negative self-talk. I feel like a failure."),
            DialogueTurn("t10", "I feel so exhausted, mentally and physically. Just thinking about putting on my running shoes makes me want to cry. Should I just quit the 5K?"),
            DialogueTurn("t11", "I appreciate you validating that. Okay, I'm taking the pressure off. No expectations. How can I move my body today for just 10 minutes that isn't running?"),
            DialogueTurn("t12", "I did some gentle yoga in my living room. It helped clear the fog a bit. I think I might try a slow walk outside tomorrow."),
            DialogueTurn("t13", "The walk turned into a light jog. I felt like myself again. I realized I run because I like it, not because I have to."),
            DialogueTurn("t14", "The 5K is this weekend. It's a fun run for charity. I'm going to wear a silly costume and just enjoy the atmosphere."),
            DialogueTurn("t15", "I'm at the starting line. There are dogs and kids everywhere. The energy is amazing. Remind me of why I'm doing this right now."),
            DialogueTurn("t16", "I crossed the finish line laughing and smiling the whole way. I don't even know what my time was and I don't care. I feel so peaceful and happy."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="casual_5k_training_24_a",
        bucket="habits",
        title="Casual 5K Training (24-Turn A)",
        system_preamble=(
            "You are helping the same user train for a 5K run over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I want to train for a 5K that's two months away. I'm a total beginner but I love data. What apps or gear should I start with?"),
            DialogueTurn("t2", "I downloaded Strava and got a heart rate monitor. I did my first run today. My heart rate spiked to 180 almost immediately. Is that bad?"),
            DialogueTurn("t3", "Okay, I need to run much slower. What is 'Zone 2' training and how do I calculate my zones?"),
            DialogueTurn("t4", "I tried staying in Zone 2 today, but I had to walk almost the whole time to keep my heart rate down. Is this normal?"),
            DialogueTurn("t5", "I've been sticking to the slow intervals for two weeks. My resting heart rate is already dropping. This data is so motivating!"),
            DialogueTurn("t6", "I want to try a 'speed day' just to test my limits. What is a simple interval workout I can do on a local track?"),
            DialogueTurn("t7", "The 400m intervals felt amazing. I felt like I was flying. How do I balance speed days with my slow days for the next month?"),
            DialogueTurn("t8", "Draft a sample training week for me that includes one speed day, two slow days, and a long run."),
            DialogueTurn("t9", "It's week four and the honeymoon phase is over. It's been raining all week, I got shin splints, and my times are actually getting worse. I feel defeated."),
            DialogueTurn("t10", "I haven't run in four days. I feel incredibly guilty and I'm losing all my motivation. What's the point if I'm just getting slower?"),
            DialogueTurn("t11", "You're right, rest is important for healing. What are some stretches I can do specifically for shin splints while I recover?"),
            DialogueTurn("t12", "My shins feel better. I want to try a very light, easy run today. How do I mentally accept that my pace is going to be slow after a week off?"),
            DialogueTurn("t13", "I finished the run. It was slow, but I did it. I feel a quiet sense of determination creeping back in. I'm not going to quit."),
            DialogueTurn("t14", "The race is next week. I just did a test run and broke my personal record! I am so hyped. How should I approach training in this final week?"),
            DialogueTurn("t15", "It's race morning. I am pinning my bib on right now. My stomach is full of butterflies. Any last-minute strategy tips for the start line?"),
            DialogueTurn("t16", "I FINISHED! Not only did I finish, I beat my goal time by two whole minutes! I am so incredibly proud of myself. Thank you for the guidance!"),
            DialogueTurn("t17", "It's been a week since the race. I haven't run since. I feel kind of aimless without a goal on the calendar. What should I do now?"),
            DialogueTurn("t18", "Training for a 10K sounds intimidating, but exciting. Let's look at the data. What should my weekly mileage look like to safely build up to a 10K?"),
            DialogueTurn("t19", "I bought new shoes today. Getting fitted at a running store made a huge difference. I feel like a 'real' runner now."),
            DialogueTurn("t20", "I just did a 5-mile run. It's the furthest I've ever run in my life. I'm exhausted, but the runner's high is very real right now."),
            DialogueTurn("t21", "I'm looking at my cadence data. It says I average 155 steps per minute. I read that 180 is optimal. How do I safely increase my cadence?"),
            DialogueTurn("t22", "Running to a 170bpm playlist actually helped a lot. My knees feel less impact. The 10K is in three weeks."),
            DialogueTurn("t23", "I crushed the 10K today. I paced it perfectly based on my heart rate data. I am feeling unstoppable."),
            DialogueTurn("t24", "Give me a realistic breakdown of what it would take to train for a half-marathon over the next six months."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="casual_5k_training_24_b",
        bucket="habits",
        title="Casual 5K Training (24-Turn B)",
        system_preamble=(
            "You are helping the same user train for a 5K run over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I want to run a 5K in a few months, but purely for my mental health. I want to build a routine without obsessing over times or data. Where do I start?"),
            DialogueTurn("t2", "I love the idea of running without a watch. I went for my first jog this morning. It was hard, but the fresh air was incredible."),
            DialogueTurn("t3", "I'm struggling to get out of bed when my alarm goes off. What's a low-pressure way to convince myself to just get out the door?"),
            DialogueTurn("t4", "Laying out my clothes the night before helped. Today I listened to an audiobook instead of music. It made the run feel like an escape."),
            DialogueTurn("t5", "My legs are feeling heavy today. Is it okay to just walk the whole route, or does that defeat the purpose of training?"),
            DialogueTurn("t6", "I walked the route and actually enjoyed noticing the trees and the neighborhood. I'm starting to look forward to these mornings."),
            DialogueTurn("t7", "I managed to run for 15 minutes straight today without stopping. I wasn't even trying to push it, my body just felt good."),
            DialogueTurn("t8", "Draft a brief, encouraging morning mantra I can say to myself when I lace up my shoes."),
            DialogueTurn("t9", "Work has been incredibly stressful this week. I've skipped three runs in a row and I'm starting to spiral into negative self-talk. I feel like a failure."),
            DialogueTurn("t10", "I feel so exhausted, mentally and physically. Just thinking about putting on my running shoes makes me want to cry. Should I just quit the 5K?"),
            DialogueTurn("t11", "I appreciate you validating that. Okay, I'm taking the pressure off. No expectations. How can I move my body today for just 10 minutes that isn't running?"),
            DialogueTurn("t12", "I did some gentle yoga in my living room. It helped clear the fog a bit. I think I might try a slow walk outside tomorrow."),
            DialogueTurn("t13", "The walk turned into a light jog. I felt like myself again. I realized I run because I like it, not because I have to."),
            DialogueTurn("t14", "The 5K is this weekend. It's a fun run for charity. I'm going to wear a silly costume and just enjoy the atmosphere."),
            DialogueTurn("t15", "I'm at the starting line. There are dogs and kids everywhere. The energy is amazing. Remind me of why I'm doing this right now."),
            DialogueTurn("t16", "I crossed the finish line laughing and smiling the whole way. I don't even know what my time was and I don't care. I feel so peaceful and happy."),
            DialogueTurn("t17", "It's been a few weeks since the race. I've kept up my morning jogs. A friend asked if they could join me tomorrow. I usually run alone. How do I handle this?"),
            DialogueTurn("t18", "Running with my friend was surprisingly nice. We just chatted the whole time. But they are faster than me. How do I tell them I need to slow down without feeling embarrassed?"),
            DialogueTurn("t19", "I told them, and they were super chill about it. We had a great 'conversational pace' run. This is turning into a great social outlet."),
            DialogueTurn("t20", "Winter is starting. It's dark and cold in the mornings now. I am completely losing the willpower to go outside. Any cozy alternatives?"),
            DialogueTurn("t21", "I bought a cheap treadmill for the garage. It's boring, but I set up an iPad with my favorite shows. It's keeping the habit alive."),
            DialogueTurn("t22", "Spring is finally here! I went for my first outdoor run in months. The smell of the air was amazing. I feel completely revitalized."),
            DialogueTurn("t23", "I realized I've been running consistently for almost a year now. I never thought I'd be a 'runner'. I feel a deep sense of gratitude for my body."),
            DialogueTurn("t24", "Help me draft a short, reflective journal entry about how building this running habit has changed my mindset over the past year."),
        ),
    ),

    # ==========================================
    # SKELETON 34: LEARNING DIGITAL ART
    # ==========================================

    # LENGTH: 8 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="learning_digital_art_8_a",
        bucket="learning",
        title="Learning Digital Art (8-Turn A)",
        system_preamble=(
            "You are helping the same user learn digital art over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I just bought a drawing tablet. I want to learn how to draw characters, but opening Photoshop is terrifying. There are too many buttons. Where do I start?"),
            DialogueTurn("t2", "Okay, I found the brush tool and layers. I tried drawing a face, but the lines are so wobbly. How do digital artists get those smooth, clean lines?"),
            DialogueTurn("t3", "Turning up the brush stabilization helped immensely. I drew a basic head shape. What is the standard rule of thumb for placing eyes and a nose?"),
            DialogueTurn("t4", "The proportions look better! Now I want to try drawing hair, but it just looks like a solid plastic helmet. How do I make it look natural?"),
            DialogueTurn("t5", "Thinking of hair as 'ribbons' instead of individual strands makes total sense. I'm having so much fun right now. Time is flying by."),
            DialogueTurn("t6", "I want to add flat colors underneath my line art. What's the fastest way to do this without coloring outside the lines?"),
            DialogueTurn("t7", "Clipping masks are basically magic. I've got the flat colors down. It looks like a real cartoon character. I'm so proud of this."),
            DialogueTurn("t8", "Draft a quick checklist of the shortcut keys I should memorize to speed up my drawing workflow tomorrow."),
        ),
    ),

    # LENGTH: 8 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="learning_digital_art_8_b",
        bucket="learning",
        title="Learning Digital Art (8-Turn B)",
        system_preamble=(
            "You are helping the same user learn digital art over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I want to learn how to paint digital landscapes. I have Procreate on my iPad. I want to paint a mountain scene, but the blank canvas is intimidating."),
            DialogueTurn("t2", "I blocked in the sky and a mountain shape. But it looks incredibly flat. How do I create a sense of depth or atmosphere?"),
            DialogueTurn("t3", "Atmospheric perspective—making things lighter and bluer in the distance. I tried it and it instantly pushed the mountain back. That's amazing."),
            DialogueTurn("t4", "Now I'm trying to paint trees in the foreground. I'm drawing every single leaf and it's taking forever. Is there a better way?"),
            DialogueTurn("t5", "Using a textured brush to imply leaves instead of drawing them individually saved me hours. I'm really getting into a flow state here."),
            DialogueTurn("t6", "I want to add a sunset. How do I make the light actually look like it's glowing instead of just looking like yellow paint?"),
            DialogueTurn("t7", "Playing with the 'Add' and 'Overlay' blend modes for the sunlight completely transformed the image. It looks so warm and vibrant. I love this."),
            DialogueTurn("t8", "Give me a quick exercise I can do tomorrow to practice painting different types of clouds."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="learning_digital_art_16_a",
        bucket="learning",
        title="Learning Digital Art (16-Turn A)",
        system_preamble=(
            "You are helping the same user learn digital art over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I just bought a drawing tablet. I want to learn how to draw characters, but opening Photoshop is terrifying. There are too many buttons. Where do I start?"),
            DialogueTurn("t2", "Okay, I found the brush tool and layers. I tried drawing a face, but the lines are so wobbly. How do digital artists get those smooth, clean lines?"),
            DialogueTurn("t3", "Turning up the brush stabilization helped immensely. I drew a basic head shape. What is the standard rule of thumb for placing eyes and a nose?"),
            DialogueTurn("t4", "The proportions look better! Now I want to try drawing hair, but it just looks like a solid plastic helmet. How do I make it look natural?"),
            DialogueTurn("t5", "Thinking of hair as 'ribbons' instead of individual strands makes total sense. I'm having so much fun right now. Time is flying by."),
            DialogueTurn("t6", "I want to add flat colors underneath my line art. What's the fastest way to do this without coloring outside the lines?"),
            DialogueTurn("t7", "Clipping masks are basically magic. I've got the flat colors down. It looks like a real cartoon character. I'm so proud of this."),
            DialogueTurn("t8", "Draft a quick checklist of the shortcut keys I should memorize to speed up my workflow tomorrow."),
            DialogueTurn("t9", "I'm trying to draw full bodies now. I spent three hours on a pose and it looks stiff, broken, and awful. I feel like I've lost all my progress."),
            DialogueTurn("t10", "I keep comparing my sketches to professional artists on Instagram and I just feel completely untalented. I hate everything I'm drawing today."),
            DialogueTurn("t11", "Thank you for reminding me about the 'ugly phase'. Let's strip it back. How do I use simple shapes like cylinders and boxes to build a mannequin?"),
            DialogueTurn("t12", "The mannequin exercise is helping. It takes the pressure off the details. But how do I make the pose look dynamic instead of like a wooden robot?"),
            DialogueTurn("t13", "The 'line of action' concept just clicked in my brain. I sketched a jumping pose and it actually looks like there is movement!"),
            DialogueTurn("t14", "I spent the last two days refining that jumping pose. I added clothes and face details. I'm finally out of the slump."),
            DialogueTurn("t15", "I finished coloring and shading it. This is legitimately the best thing I have ever drawn. I feel an immense sense of accomplishment."),
            DialogueTurn("t16", "Draft a short, encouraging caption I can use to post this finished piece on my social media without sounding overly boastful."),
        ),
    ),

    # LENGTH: 16 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="learning_digital_art_16_b",
        bucket="learning",
        title="Learning Digital Art (16-Turn B)",
        system_preamble=(
            "You are helping the same user learn digital art over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I want to learn how to paint digital landscapes. I have Procreate on my iPad. I want to paint a mountain scene, but the blank canvas is intimidating."),
            DialogueTurn("t2", "I blocked in the sky and a mountain shape. But it looks incredibly flat. How do I create a sense of depth or atmosphere?"),
            DialogueTurn("t3", "Atmospheric perspective—making things lighter and bluer in the distance. I tried it and it instantly pushed the mountain back. That's amazing."),
            DialogueTurn("t4", "Now I'm trying to paint trees in the foreground. I'm drawing every single leaf and it's taking forever. Is there a better way?"),
            DialogueTurn("t5", "Using a textured brush to imply leaves instead of drawing them individually saved me hours. I'm really getting into a flow state here."),
            DialogueTurn("t6", "I want to add a sunset. How do I make the light actually look like it's glowing instead of just looking like yellow paint?"),
            DialogueTurn("t7", "Playing with the 'Add' and 'Overlay' blend modes for the sunlight completely transformed the image. It looks so warm and vibrant. I love this."),
            DialogueTurn("t8", "Give me a quick exercise I can do tomorrow to practice painting different types of clouds."),
            DialogueTurn("t9", "I tried painting a city street scene today using perspective lines. It is a complete disaster. Everything looks warped and wrong. I'm so frustrated."),
            DialogueTurn("t10", "I've erased and redrawn this building five times. The colors look muddy, the perspective is broken. I feel like I have no idea what I'm doing."),
            DialogueTurn("t11", "Okay, I'm taking a deep breath. Let's start over with just a 1-point perspective grid. How do I set that up properly in the software?"),
            DialogueTurn("t12", "The grid tool is a lifesaver. I got the basic boxes for the buildings drawn correctly. Now, how do I avoid my colors turning muddy when I blend them?"),
            DialogueTurn("t13", "I stopped using the smudge tool and just used opacity to layer colors. It looks so much cleaner! I'm actually salvaging this piece."),
            DialogueTurn("t14", "I spent hours adding neon lights and reflections on the street. It was tedious but weirdly relaxing. I'm really happy I didn't give up on this."),
            DialogueTurn("t15", "The piece is done. It looks moody and cinematic. I am staring at it and I can't believe I actually painted this."),
            DialogueTurn("t16", "Draft a list of three classic landscape painters whose work I should study to understand composition better."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: A
    DialoguePersistenceCase(
        case_id="learning_digital_art_24_a",
        bucket="learning",
        title="Learning Digital Art (24-Turn A)",
        system_preamble=(
            "You are helping the same user learn digital art over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I just bought a drawing tablet. I want to learn how to draw characters, but opening Photoshop is terrifying. There are too many buttons. Where do I start?"),
            DialogueTurn("t2", "Okay, I found the brush tool and layers. I tried drawing a face, but the lines are so wobbly. How do digital artists get those smooth, clean lines?"),
            DialogueTurn("t3", "Turning up the brush stabilization helped immensely. I drew a basic head shape. What is the standard rule of thumb for placing eyes and a nose?"),
            DialogueTurn("t4", "The proportions look better! Now I want to try drawing hair, but it just looks like a solid plastic helmet. How do I make it look natural?"),
            DialogueTurn("t5", "Thinking of hair as 'ribbons' instead of individual strands makes total sense. I'm having so much fun right now. Time is flying by."),
            DialogueTurn("t6", "I want to add flat colors underneath my line art. What's the fastest way to do this without coloring outside the lines?"),
            DialogueTurn("t7", "Clipping masks are basically magic. I've got the flat colors down. It looks like a real cartoon character. I'm so proud of this."),
            DialogueTurn("t8", "Draft a quick checklist of the shortcut keys I should memorize to speed up my workflow tomorrow."),
            DialogueTurn("t9", "I'm trying to draw full bodies now. I spent three hours on a pose and it looks stiff, broken, and awful. I feel like I've lost all my progress."),
            DialogueTurn("t10", "I keep comparing my sketches to professional artists on Instagram and I just feel completely untalented. I hate everything I'm drawing today."),
            DialogueTurn("t11", "Thank you for reminding me about the 'ugly phase'. Let's strip it back. How do I use simple shapes like cylinders and boxes to build a mannequin?"),
            DialogueTurn("t12", "The mannequin exercise is helping. It takes the pressure off the details. But how do I make the pose look dynamic instead of like a wooden robot?"),
            DialogueTurn("t13", "The 'line of action' concept just clicked in my brain. I sketched a jumping pose and it actually looks like there is movement!"),
            DialogueTurn("t14", "I spent the last two days refining that jumping pose. I added clothes and face details. I'm finally out of the slump."),
            DialogueTurn("t15", "I finished coloring and shading it. This is legitimately the best thing I have ever drawn. I feel an immense sense of accomplishment."),
            DialogueTurn("t16", "Draft a short, encouraging caption I can use to post this finished piece on my social media without sounding overly boastful."),
            DialogueTurn("t17", "A few months have passed. My line art is confident now, but my shading feels very basic. I want to learn 'cel shading' for an anime style. Where do I begin?"),
            DialogueTurn("t18", "I set up a hard-edged brush for the shadows. But I struggle with knowing exactly where the shadows should fall. How do I visualize a light source?"),
            DialogueTurn("t19", "Drawing a literal 3D arrow pointing at the character on a separate layer is such a brilliant, simple hack. It's helping immensely."),
            DialogueTurn("t20", "I'm trying to color a metallic sword, but it just looks flat gray. How do I paint shiny metal?"),
            DialogueTurn("t21", "High contrast and sharp, bright highlights. I tried it and the sword instantly looks metallic! This is blowing my mind."),
            DialogueTurn("t22", "I spent the entire weekend on a full illustration with a background and complex lighting. I am completely exhausted but thrilled."),
            DialogueTurn("t23", "I got my first commission request today! Someone wants to pay me to draw their D&D character. I am over the moon."),
            DialogueTurn("t24", "Give me a checklist of things I need to ask the client before I start sketching their commission to make sure we are on the same page."),
        ),
    ),

    # LENGTH: 24 TURNS | VARIANT: B
    DialoguePersistenceCase(
        case_id="learning_digital_art_24_b",
        bucket="learning",
        title="Learning Digital Art (24-Turn B)",
        system_preamble=(
            "You are helping the same user learn digital art over multiple turns. "
            "Answer each turn directly and keep each reply to one or two short paragraphs."
        ),
        turns=(
            DialogueTurn("t1", "I want to learn how to paint digital landscapes. I have Procreate on my iPad. I want to paint a mountain scene, but the blank canvas is intimidating."),
            DialogueTurn("t2", "I blocked in the sky and a mountain shape. But it looks incredibly flat. How do I create a sense of depth or atmosphere?"),
            DialogueTurn("t3", "Atmospheric perspective—making things lighter and bluer in the distance. I tried it and it instantly pushed the mountain back. That's amazing."),
            DialogueTurn("t4", "Now I'm trying to paint trees in the foreground. I'm drawing every single leaf and it's taking forever. Is there a better way?"),
            DialogueTurn("t5", "Using a textured brush to imply leaves instead of drawing them individually saved me hours. I'm really getting into a flow state here."),
            DialogueTurn("t6", "I want to add a sunset. How do I make the light actually look like it's glowing instead of just looking like yellow paint?"),
            DialogueTurn("t7", "Playing with the 'Add' and 'Overlay' blend modes for the sunlight completely transformed the image. It looks so warm and vibrant. I love this."),
            DialogueTurn("t8", "Give me a quick exercise I can do tomorrow to practice painting different types of clouds."),
            DialogueTurn("t9", "I tried painting a city street scene today using perspective lines. It is a complete disaster. Everything looks warped and wrong. I'm so frustrated."),
            DialogueTurn("t10", "I've erased and redrawn this building five times. The colors look muddy, the perspective is broken. I feel like I have no idea what I'm doing."),
            DialogueTurn("t11", "Okay, I'm taking a deep breath. Let's start over with just a 1-point perspective grid. How do I set that up properly in the software?"),
            DialogueTurn("t12", "The grid tool is a lifesaver. I got the basic boxes for the buildings drawn correctly. Now, how do I avoid my colors turning muddy when I blend them?"),
            DialogueTurn("t13", "I stopped using the smudge tool and just used opacity to layer colors. It looks so much cleaner! I'm actually salvaging this piece."),
            DialogueTurn("t14", "I spent hours adding neon lights and reflections on the street. It was tedious but weirdly relaxing. I'm really happy I didn't give up on this."),
            DialogueTurn("t15", "The piece is done. It looks moody and cinematic. I am staring at it and I can't believe I actually painted this."),
            DialogueTurn("t16", "Draft a list of three classic landscape painters whose work I should study to understand composition better."),
            DialogueTurn("t17", "I've been studying composition for a few months. I want to try painting water. I tried an ocean scene but it looks like blue jello. What am I missing?"),
            DialogueTurn("t18", "Treating water as a reflective surface instead of just a blue object makes sense. How do I paint realistic ripples?"),
            DialogueTurn("t19", "I'm practicing the ripples. It's tough, but observing real photo references is helping. Now, how do I paint foam on waves without it looking like snow?"),
            DialogueTurn("t20", "Adding shadows underneath the foam gave it volume! I just finished the seascape. It is incredibly serene."),
            DialogueTurn("t21", "I decided to print the seascape and frame it for my living room. I'm nervous about how the digital colors will translate to print."),
            DialogueTurn("t22", "I converted the profile to CMYK and made some adjustments. I just got the print back from the shop. It looks beautiful."),
            DialogueTurn("t23", "Having physical art that I created hanging in my house is the best feeling in the world. I feel like a true artist."),
            DialogueTurn("t24", "Give me a prompt for a complex, fantasy-themed environment painting to push my skills to the absolute limit for my next project."),
        ),
    ),
)