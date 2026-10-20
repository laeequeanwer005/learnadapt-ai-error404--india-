from datetime import datetime
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="LearnAdapt AI | Learning Intelligence", page_icon="L", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--navy:#12213f;--blue:#2f65f5;--ink:#18243b;--muted:#758198;--line:#e5eaf3;--canvas:#f5f7fb}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:var(--canvas);color:var(--ink)}
[data-testid="stHeader"]{background:rgba(245,247,251,.92)}
[data-testid="stSidebar"]{background:#fff;border-right:1px solid var(--line)}
[data-testid="stSidebar"]>div:first-child{padding:1.1rem .9rem}
.block-container{padding-top:2rem;padding-bottom:2.5rem;max-width:1500px}
.brand{display:flex;align-items:center;gap:12px;padding:5px 7px 24px}
.brand-logo{width:44px;height:44px;border-radius:15px;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#2357e7,#71a4ff);color:white;font-family:Manrope,sans-serif;font-weight:800;font-size:23px;box-shadow:0 8px 20px #2f65f53d}
.brand-title{font-family:Manrope,sans-serif;font-size:17px;font-weight:800;color:var(--navy);line-height:1.2}
.brand-sub{font-size:11px;color:var(--muted);margin-top:4px}
.nav-caption,.eyebrow{font-size:10px;font-weight:800;letter-spacing:.15em;color:var(--blue);text-transform:uppercase;margin-bottom:9px}
.page-title{font-family:Manrope,sans-serif;font-size:clamp(27px,3vw,37px);font-weight:800;letter-spacing:-.045em;line-height:1.14;color:var(--navy);margin:0}
.page-subtitle{font-size:14px;line-height:1.7;color:var(--muted);margin:10px 0 18px;max-width:780px}
.top-meta{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:14px}
.pill{display:inline-block;padding:5px 10px;border-radius:999px;font-size:10px;font-weight:800;letter-spacing:.04em}
.pill-blue{background:#edf3ff;color:#2458d7;border:1px solid #d9e5ff}.pill-amber{background:#fff7e7;color:#946200;border:1px solid #f5dfad}.pill-green{background:#e8f8f4;color:#087b70;border:1px solid #c7eee5}
.panel{background:white;border:1px solid var(--line);border-radius:18px;padding:20px 21px;box-shadow:0 5px 20px #13264a09;height:100%}
.panel-title{font-family:Manrope,sans-serif;font-size:16px;font-weight:800;color:var(--navy);letter-spacing:-.025em;margin:0}
.panel-sub{font-size:11px;color:var(--muted);margin-top:5px;margin-bottom:16px}
.hero{border-radius:22px;padding:26px 27px;min-height:220px;background:linear-gradient(115deg,#173d9b 0%,#2861ee 58%,#76a5ff 100%);color:white;box-shadow:0 15px 34px #2f65f52b}
.hero-kicker{font-size:10px;font-weight:800;letter-spacing:.16em;color:#dbe7ff}
.hero-title{font-family:Manrope,sans-serif;font-size:clamp(23px,2.4vw,31px);font-weight:800;line-height:1.18;letter-spacing:-.04em;margin-top:14px}
.hero-copy{font-size:12px;line-height:1.75;color:#e6eeff;margin-top:11px;max-width:560px}
.profile-panel{background:#fff;border:1px solid var(--line);border-radius:18px;padding:22px;height:100%}
.profile-avatar{width:43px;height:43px;border-radius:14px;background:#edf3ff;color:#2458d7;display:flex;align-items:center;justify-content:center;font-size:16px;font-weight:800}
.profile-copy{font-size:11px;color:var(--muted);line-height:1.7;margin-top:13px}
.metric{background:#fff;border:1px solid var(--line);border-radius:17px;padding:17px 18px;box-shadow:0 4px 16px #13264a06;min-height:124px}
.metric-top{display:flex;align-items:center;justify-content:space-between;gap:8px}.metric-label{font-size:11px;font-weight:600;color:var(--muted)}
.metric-icon{width:33px;height:33px;border-radius:11px;display:flex;align-items:center;justify-content:center;background:#edf3ff;color:var(--blue);font-weight:800}
.metric-value{font-family:Manrope,sans-serif;font-size:27px;line-height:1.2;font-weight:800;letter-spacing:-.04em;color:var(--navy);margin-top:9px}
.metric-foot{font-size:10px;color:#939bad;margin-top:6px}
.callout{border:1px dashed #bfd0f2;background:#f8faff;border-radius:14px;padding:15px 16px;color:#596982;font-size:12px;line-height:1.75}
.activity-title{font-family:Manrope,sans-serif;font-size:21px;font-weight:800;color:var(--navy);letter-spacing:-.03em;margin:8px 0}
.muted{color:var(--muted);font-size:12px;line-height:1.7}.small-label{font-size:10px;font-weight:800;letter-spacing:.1em;color:#909bb0;text-transform:uppercase}
.sidebar-team{border-top:1px solid var(--line);margin-top:22px;padding:19px 8px 0}
.footer{border-top:1px solid var(--line);margin-top:28px;padding-top:16px;color:#9aa3b3;font-size:10px;line-height:1.8}
div[data-testid="stMetric"]{background:white;border:1px solid var(--line);border-radius:16px;padding:15px 18px}
div[data-testid="stMetricLabel"] p{font-size:11px;color:var(--muted)} div[data-testid="stMetricValue"]{font-family:Manrope,sans-serif;font-weight:800;color:var(--navy)}
.stButton>button,.stFormSubmitButton>button{border-radius:10px;font-weight:700;min-height:42px}
div[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:13px;overflow:hidden}
@media(max-width:800px){.block-container{padding:1.2rem 1rem 2rem}.panel,.profile-panel{padding:16px}.hero{padding:22px;min-height:unset}}
</style>
""", unsafe_allow_html=True)

SKILLS = [
    {"skill":"Fractions","progress":72,"category":"Foundations"},
    {"skill":"Algebra foundations","progress":58,"category":"Core concepts"},
    {"skill":"Linear graphs","progress":44,"category":"Applied skills"},
    {"skill":"Equations","progress":63,"category":"Core concepts"},
]
SAMPLE_HISTORY = [
    {"timestamp":"2026-10-20 10:15","skill":"Fractions","activity":"Fraction operations","result":"Correct","score":"4 / 5","duration":"8 min"},
    {"timestamp":"2026-10-20 09:42","skill":"Algebra foundations","activity":"One-step equations","result":"Incorrect","score":"2 / 5","duration":"6 min"},
    {"timestamp":"2026-10-19 18:10","skill":"Linear graphs","activity":"Coordinate plane basics","result":"Correct","score":"5 / 5","duration":"7 min"},
    {"timestamp":"2026-10-19 17:35","skill":"Fractions","activity":"Equivalent fractions","result":"Correct","score":"5 / 5","duration":"5 min"},
    {"timestamp":"2026-10-18 16:20","skill":"Equations","activity":"Balance equations","result":"Incorrect","score":"2 / 5","duration":"9 min"},
]
if "history" not in st.session_state:
    st.session_state.history = SAMPLE_HISTORY.copy()

def header(eyebrow, title, subtitle):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div><h1 class="page-title">{title}</h1><div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)

def panel_heading(title, subtitle=""):
    st.markdown(f'<div class="panel-title">{title}</div><div class="panel-sub">{subtitle}</div>', unsafe_allow_html=True)

def metric_card(label, value, foot, icon):
    st.markdown(f'<div class="metric"><div class="metric-top"><div class="metric-label">{label}</div><div class="metric-icon">{icon}</div></div><div class="metric-value">{value}</div><div class="metric-foot">{foot}</div></div>', unsafe_allow_html=True)

def demo_note():
    st.markdown('<div class="top-meta"><span class="pill pill-amber">DEMO DATA</span><span class="muted">Synthetic sample values · session-only updates · no permanent storage</span></div>', unsafe_allow_html=True)

def skill_chart():
    df = pd.DataFrame(SKILLS).sort_values("progress")
    fig = px.bar(df, x="progress", y="skill", orientation="h", text="progress", color="skill",
                 color_discrete_sequence=["#2f65f5","#12998c","#8b5cf6","#e7a12a"])
    fig.update_traces(texttemplate="%{text}%", textposition="outside", cliponaxis=False, marker_line_width=0)
    fig.update_layout(height=290,margin=dict(l=0,r=30,t=8,b=5),showlegend=False,
        xaxis=dict(range=[0,110],ticksuffix="%",title=None,showgrid=True,gridcolor="#edf1f7",zeroline=False),
        yaxis=dict(title=None,showgrid=False,automargin=True),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans",color="#758198",size=11))
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

def history_table(rows, limit=None):
    df = pd.DataFrame(rows)
    if df.empty:
        st.info("No practice entries yet. Add one from Practice Update.")
        return
    if limit: df = df.head(limit)
    df = df.rename(columns={"timestamp":"Timestamp","skill":"Skill","activity":"Activity","result":"Result","score":"Score","duration":"Duration"})
    st.dataframe(df,use_container_width=True,hide_index=True)

with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-logo">L</div><div><div class="brand-title">LearnAdapt AI</div><div class="brand-sub">Personalized learning paths</div></div></div><div class="nav-caption">Workspace</div>',unsafe_allow_html=True)
    page = st.radio("Workspace navigation",["Dashboard","Skill Progress","Practice History","Next Learning Activity","Practice Update"],label_visibility="collapsed")
    st.markdown('<div class="sidebar-team"><div style="font-weight:800;font-size:12px;color:#17213b;">TEAM ERROR-404</div><div style="font-size:11px;color:#8791a4;margin-top:7px;">AI Innovation Hackathon 2026</div><div style="margin-top:15px;padding:13px;border:1px dashed #cbd8f2;border-radius:13px;background:#f8faff;"><div style="font-size:11px;font-weight:800;color:#2f65f5;">FRONTEND PREVIEW</div><div style="font-size:11px;line-height:1.6;color:#758198;margin-top:5px;">Model and recommendation backend are not connected.</div></div></div>',unsafe_allow_html=True)

if page == "Dashboard":
    header("LEARNING INTELLIGENCE","Every learner has a different starting point.","A focused workspace to review skill progress, understand practice patterns, and prepare for a more personalized learning journey.")
    demo_note()
    hero_col, profile_col = st.columns([1.55,1],gap="large")
    with hero_col:
        st.markdown('<div class="hero"><div class="hero-kicker">LEARNADAPT LEARNING SPACE</div><div class="hero-title">Make the next learning step<br/>more intentional.</div><div class="hero-copy">Explore a learner’s progress and practice history in one place. LearnAdapt AI is designed to make future recommendations transparent, explainable, and responsive to learning evidence.</div></div>',unsafe_allow_html=True)
    with profile_col:
        st.markdown('<div class="profile-panel"><div class="small-label">CURRENT WORKSPACE</div><div style="display:flex;align-items:center;gap:12px;margin-top:13px;"><div class="profile-avatar">DA</div><div><div style="font-family:Manrope;font-size:17px;font-weight:800;color:#12213f;">Demo learner A</div><div style="font-size:11px;color:#758198;margin-top:3px;">Synthetic profile · Preview mode</div></div></div><div style="margin-top:18px;"><span class="pill pill-green">● Sample profile loaded</span></div><div class="profile-copy">No real student records are used. Values are illustrative and do not represent model predictions.</div></div>',unsafe_allow_html=True)
    st.write("")
    avg = round(sum(x["progress"] for x in SKILLS)/len(SKILLS))
    cols = st.columns(4,gap="medium")
    for col,args in zip(cols,[("Skills tracked","04","Synthetic skill set","◇"),("Practice attempts",f"{len(st.session_state.history):02d}","Sample + session entries","✓"),("Average progress",f"{avg}%","Illustrative values only","↗"),("Model status","Pending","Backend integration","⌁")]):
        with col: metric_card(*args)
    st.write("")
    chart_col, history_col = st.columns([1.08,1],gap="large")
    with chart_col:
        with st.container(border=True):
            panel_heading("Skill progress","Illustrative values across the sample skill set")
            skill_chart()
    with history_col:
        with st.container(border=True):
            panel_heading("Recent practice","Latest entries from the sample practice log")
            history_table(st.session_state.history,4)
    with st.container(border=True):
        panel_heading("Next learning activity","A dedicated space for the future recommendation service")
        a,b = st.columns([1,1.5],gap="large")
        with a: st.markdown('<span class="pill pill-amber">PLACEHOLDER · NOT AN AI PREDICTION</span><div class="activity-title">Your next activity</div><div class="muted">Activity: Not assigned<br/>Skill: Not assigned<br/>Difficulty: Not assigned</div>',unsafe_allow_html=True)
        with b: st.markdown('<div class="callout"><b>Backend integration pending</b><br/>When the recommendation service is implemented, this panel can display an activity and a reason based on available learning evidence. The current frontend does not generate predictions.</div>',unsafe_allow_html=True)

elif page == "Skill Progress":
    header("SKILL ANALYTICS","Skill progress","A clear overview of the sample skill set. Values are synthetic and are not predicted mastery.")
    demo_note()
    with st.container(border=True):
        panel_heading("Progress by skill","Compare illustrative progress across foundations and applied concepts")
        skill_chart()
    c1,c2 = st.columns(2,gap="medium")
    for i,s in enumerate(SKILLS):
        with [c1,c2][i%2]:
            with st.container(border=True):
                top = st.columns([3,1]); top[0].markdown(f"**{s['skill']}**"); top[1].markdown(f"**{s['progress']}%**")
                st.progress(s["progress"]/100); st.caption(f"{s['category']} · Synthetic sample value")
    st.markdown('<div class="callout"><b>Integration note</b><br/>Replace static sample values with verified skill data from the future data-processing module. Keep the synthetic label until then.</div>',unsafe_allow_html=True)

elif page == "Practice History":
    header("PRACTICE JOURNAL","Practice history","Review practice activity, outcomes, and time spent. Sample timestamps are illustrative.")
    demo_note()
    rows=st.session_state.history; total=len(rows); correct=sum(r["result"].lower()=="correct" for r in rows)
    k1,k2,k3=st.columns(3); k1.metric("Attempts shown",total); k2.metric("Correct results",correct); k3.metric("Displayed-data accuracy",f"{round(correct/total*100) if total else 0}%")
    with st.container(border=True):
        panel_heading("All practice attempts","Filter by skill to inspect activity")
        choices=["All skills"]+[s["skill"] for s in SKILLS]; filter_skill=st.selectbox("Filter by skill",choices)
        history_table(rows if filter_skill=="All skills" else [r for r in rows if r["skill"]==filter_skill])
    st.caption("Displayed-data accuracy is calculated from sample/session records only; it is not a learner performance claim.")

elif page == "Next Learning Activity":
    header("LEARNING PATH","Next learning activity","A polished placeholder for the future explainable recommendation experience.")
    st.warning("Placeholder only — no ML model or recommendation algorithm is connected. No activity is being predicted.")
    left,right=st.columns([1.15,1],gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<span class="pill pill-amber">AWAITING BACKEND</span><div class="activity-title">Activity details will appear here</div><div class="muted">Skill: Not assigned<br/>Difficulty: Not assigned<br/>Estimated duration: Not assigned</div>',unsafe_allow_html=True)
            st.button("Start activity",disabled=True,use_container_width=True)
    with right:
        with st.container(border=True):
            panel_heading("Why this activity?","Explanation placeholder")
            st.markdown('<div class="callout">When implemented, the recommendation service can provide an explanation grounded in learner evidence and prerequisite rules. No explanation is generated by this frontend.</div>',unsafe_allow_html=True)
            st.write(""); panel_heading("Future response format","Illustrative integration contract")
            st.code('{\n  "activity_title": "...",\n  "skill": "...",\n  "difficulty": "...",\n  "estimated_minutes": 10,\n  "explanation": "..."\n}',language="json")

elif page == "Practice Update":
    header("PRACTICE ENTRY","Update practice","Record a sample attempt to test the interaction flow. Entries are temporary and session-only.")
    st.info("No permanent database is connected. Entries may disappear when the app session resets.")
    with st.container(border=True):
        panel_heading("Record a practice attempt","Enter the activity details below")
        with st.form("practice_form",clear_on_submit=True):
            left,right=st.columns(2)
            with left:
                skill=st.selectbox("Skill",[s["skill"] for s in SKILLS])
                result=st.selectbox("Result",["Correct","Incorrect","Partially correct"])
                score=st.selectbox("Score",["1 / 5","2 / 5","3 / 5","4 / 5","5 / 5"])
            with right:
                activity=st.text_input("Activity name",placeholder="e.g. Equivalent fractions",max_chars=100)
                duration=st.number_input("Duration (minutes)",min_value=1,max_value=240,value=10)
                notes=st.text_area("Optional notes",placeholder="What did you practice?",max_chars=500)
            submit=st.form_submit_button("Add practice entry",type="primary",use_container_width=True)
        if submit:
            if not activity.strip(): st.error("Enter an activity name before submitting.")
            else:
                st.session_state.history.insert(0,{"timestamp":datetime.now().strftime("%Y-%m-%d %H:%M"),"skill":skill,"activity":activity.strip(),"result":result,"score":score,"duration":f"{int(duration)} min"})
                st.success("Practice entry added to this session. It has not been saved permanently.")
                st.rerun()
    with st.container(border=True):
        panel_heading("Latest entries","Recent session activity")
        history_table(st.session_state.history,5)

st.markdown('<div class="footer"><b style="color:#71809a;">LearnAdapt AI</b> · TEAM ERROR-404 · AI Innovation Hackathon 2026<br/>Frontend prototype · Synthetic data · Recommendation backend not connected · No permanent storage</div>',unsafe_allow_html=True)
