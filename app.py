import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import defaultdict

st.set_page_config(page_title="World Cup 2026 Predictor", page_icon="⚽", layout="wide")

# ---------------- UI THEME (lightweight football vibe) ----------------
FOOTBALL_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Bungee&family=Inter:wght@400;600;700&display=swap');
  html, body, [class*="css"] { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
  .pitch {
    background:
      radial-gradient(circle at 10% 20%, rgba(204,255,204,.8) 0 12%, transparent 13% 100%),
      radial-gradient(circle at 80% 10%, rgba(204,255,204,.7) 0 12%, transparent 13% 100%),
      radial-gradient(circle at 80% 80%, rgba(204,255,204,.8) 0 10%, transparent 11% 100%),
      linear-gradient(90deg, #2b8a3e 0 50%, #2f9e44 50% 100%);
    background-size: cover;
    padding-bottom: 1rem;
    min-height: 100vh;
  }
  .card { background:#fff; border:3px solid #0b3; border-radius:16px; box-shadow:6px 6px 0 rgba(0,0,0,.15); padding:1rem; margin-bottom:1rem; }
  .title { font-family: Bungee, cursive; letter-spacing:1px; display:inline-block; background:#fff; border:4px solid #0b3; border-radius:12px; padding:.4rem .8rem; box-shadow:6px 6px 0 rgba(0,0,0,.2); }
  .pill { display:inline-block; background:#e6ffed; border:2px solid #2f9e44; border-radius:999px; padding:.1rem .6rem; margin:.1rem; font-weight:700; }
  .subtle { opacity:.8; font-size:.9rem; }
  .grid { display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap:.6rem;}
  .groupbox { background:#f7fff9; border:2px dashed #0b3; border-radius:12px; padding:.6rem; }
</style>
"""
st.markdown('<div class="pitch">', unsafe_allow_html=True)
st.markdown(FOOTBALL_CSS, unsafe_allow_html=True)
st.markdown('<h1 class="title">FIFA World Cup 2026 — Simple Predictor</h1>', unsafe_allow_html=True)
st.caption("Toy simulator for fun. Uses ratings to play out group & knockout rounds.")

# ---------------- Data loading ----------------
@st.cache_data
def load_teams():
    df = pd.read_csv("teams.csv")
    # Ensure required cols
    need = {"Team","Confed","Rating"}
    if not need.issubset(df.columns):
        raise ValueError("teams.csv must include columns: Team, Confed, Rating")
    # Limit to top 48 by rating (in case file has more)
    df = df.sort_values("Rating", ascending=False).head(48).reset_index(drop=True)
    return df

def write_default_csv():
    default_csv = """Team,Confed,Rating
    Argentina,CONMEBOL,95
    France,UEFA,94
    Brazil,CONMEBOL,93
    England,UEFA,92
    Spain,UEFA,92
    Portugal,UEFA,91
    Netherlands,UEFA,90
    Germany,UEFA,90
    Italy,UEFA,89
    Belgium,UEFA,89
    Croatia,UEFA,88
    Uruguay,CONMEBOL,87
    Colombia,CONMEBOL,86
    Morocco,CAF,86
    USA,CONCACAF,85
    Mexico,CONCACAF,84
    Denmark,UEFA,84
    Switzerland,UEFA,84
    Japan,AFC,84
    Austria,UEFA,83
    Senegal,CAF,83
    Nigeria,CAF,82
    Ukraine,UEFA,82
    Poland,UEFA,81
    Serbia,UEFA,81
    Romania,UEFA,80
    Hungary,UEFA,80
    Czechia,UEFA,79
    South Korea,AFC,79
    Ecuador,CONMEBOL,79
    Australia,AFC,78
    Scotland,UEFA,78
    Iran,AFC,78
    Turkey,UEFA,78
    Algeria,CAF,77
    Tunisia,CAF,77
    Canada,CONCACAF,77
    Ghana,CAF,76
    Egypt,CAF,76
    Panama,CONCACAF,75
    Costa Rica,CONCACAF,75
    Norway,UEFA,75
    Slovakia,UEFA,74
    Saudi Arabia,AFC,74
    Cameroon,CAF,74
    Paraguay,CONMEBOL,74
    Qatar,AFC,73
    China PR,AFC,70
    India,AFC,67
    """.strip()
    with open("teams.csv","w",encoding="utf-8") as f:
        f.write(default_csv)

try:
    teams = load_teams()
except Exception as e:
    st.info("No valid teams.csv found — writing a default one with sample ratings.")
    write_default_csv()
    teams = load_teams()

st.sidebar.header("⚙️ Simulator Settings")
sims = st.sidebar.slider("Number of simulations", 100, 5000, 1000, step=100)
draw_bias = st.sidebar.slider("Group-stage draw rate (baseline %)", 0, 40, 24)
upset_factor = st.sidebar.slider("Upset factor (higher = more randomness)", 0.0, 2.0, 1.0, step=0.1)
seed = st.sidebar.text_input("Random seed (optional)", "")
avoid_same_group_r32 = st.sidebar.checkbox("Avoid same-group matches in R32 (best effort)", True)

# Allow uploading a custom teams.csv
up = st.sidebar.file_uploader("Upload custom teams.csv", type=["csv"])
if up is not None:
    teams = pd.read_csv(up)
    st.sidebar.success("Loaded uploaded teams list.")
    # Limit to max 48
    teams = teams.sort_values("Rating", ascending=False).head(48).reset_index(drop=True)

# ---------------- Helpers ----------------
rng = np.random.default_rng(abs(hash(seed)) % (2**32)) if seed else np.random.default_rng()

GROUP_LABELS = list("ABCDEFGHIJKL")  # 12 groups of 4

def seed_groups(df: pd.DataFrame) -> dict:
    # Pot seeding by rating into 12 groups x 4
    df = df.sort_values("Rating", ascending=False).reset_index(drop=True)
    pots = [df.iloc[i*12:(i+1)*12] for i in range(4)]
    groups = {g: [] for g in GROUP_LABELS}
    for p in pots:
        idxs = list(range(12))
        rng.shuffle(idxs)
        for g_idx, row in zip(idxs, p.itertuples(index=False)):
            groups[GROUP_LABELS[g_idx]].append((row.Team, row.Confed, float(row.Rating)))
    return groups

def bt_win_prob(r1, r2):
    # Bradley-Terry logistic; upset_factor raises randomness
    k = 8.0 / max(1e-9, upset_factor + 0.5)  # scale
    return 1.0 / (1.0 + np.exp(-(r1 - r2)/k))

def simulate_group(groups):
    # Round-robin matches within each group (3 games per team)
    standings = {}
    for g, teams4 in groups.items():
        # per-team stats
        stats = {t[0]: {"pts":0, "gf":0, "ga":0, "gd":0, "rating":t[2], "confed":t[1]} for t in teams4}
        # all pairings
        for i in range(4):
            for j in range(i+1,4):
                t1, c1, r1 = teams4[i][0], teams4[i][1], teams4[i][2]
                t2, c2, r2 = teams4[j][0], teams4[j][1], teams4[j][2]
                p1 = bt_win_prob(r1, r2)
                p2 = 1 - p1
                draw_p = draw_bias/100.0
                # Normalize (win1, draw, win2)
                p1n = p1*(1-draw_p)
                p2n = p2*(1-draw_p)
                pn = p1n + p2n + draw_p
                p1n, draw_pn, p2n = p1n/pn, draw_p/pn, p2n/pn
                r = rng.random()
                # simple goal model (0-4 goals)
                g1 = int(np.clip(np.round(max(0, r1-65)/10 + rng.normal(1.2, 0.9)), 0, 4))
                g2 = int(np.clip(np.round(max(0, r2-65)/10 + rng.normal(1.2, 0.9)), 0, 4))
                # flip outcome to enforce result distribution
                if r < p1n:
                    if g1 <= g2: g1 = g2 + 1
                    stats[t1]["pts"] += 3
                elif r < p1n + draw_pn:
                    g2 = g1  # draw
                    stats[t1]["pts"] += 1; stats[t2]["pts"] += 1
                else:
                    if g2 <= g1: g2 = g1 + 1
                    stats[t2]["pts"] += 3
                stats[t1]["gf"] += g1; stats[t1]["ga"] += g2; stats[t1]["gd"] = stats[t1]["gf"]-stats[t1]["ga"]
                stats[t2]["gf"] += g2; stats[t2]["ga"] += g1; stats[t2]["gd"] = stats[t2]["gf"]-stats[t2]["ga"]
        # rank by pts, gd, gf, rating (proxy), random tiebreak
        order = sorted(stats.items(), key=lambda kv: (kv[1]["pts"], kv[1]["gd"], kv[1]["gf"], kv[1]["rating"], rng.random()), reverse=True)
        standings[g] = order
    return standings

def pick_knockout(standings):
    firsts, seconds, thirds = [], [], []
    for g in GROUP_LABELS:
        ordered = [t for t,_ in standings[g]]
        firsts.append(ordered[0])
        seconds.append(ordered[1])
        thirds.append(ordered[2])
    # Best 8 thirds by points/gd/gf
    third_stats = []
    for g in GROUP_LABELS:
        t, s = standings[g][2]
        third_stats.append((t, s["pts"], s["gd"], s["gf"], g))
    third_qual = sorted(third_stats, key=lambda x: (x[1], x[2], x[3], rng.random()), reverse=True)[:8]
    third_names = [t for t,_,_,_,_ in third_qual]

    # Build Round of 32 pairings (simple heuristic bracketing)
    # 1st vs 3rd if possible, else 1st vs a 2nd not from same group
    r32 = []
    available_seconds = seconds.copy()
    available_thirds = third_names.copy()
    # Pair winners first
    for idx, g in enumerate(GROUP_LABELS):
        w = firsts[idx]
        # try third not from same group
        opp = None
        for t in list(available_thirds):
            # get group of t
            grp_t = next(G for G in GROUP_LABELS if t in [name for name,_ in standings[G]])
            if not avoid_same_group_r32 or grp_t != g:
                opp = t; break
        if opp is not None:
            available_thirds.remove(opp)
            r32.append((w, opp))
        else:
            # pick a second not from same group
            for s in list(available_seconds):
                grp_s = next(G for G in GROUP_LABELS if s in [name for name,_ in standings[G]])
                if grp_s != g:
                    opp = s; break
            if opp is None and available_seconds:
                opp = available_seconds[0]
            if opp:
                available_seconds.remove(opp)
                r32.append((w, opp))
    # Pair whatever remains among seconds/thirds
    pool = available_seconds + available_thirds
    rng.shuffle(pool)
    while len(pool) >= 2:
        a = pool.pop(); b = pool.pop()
        r32.append((a,b))

    # If under/overflow due to pathologies, trim/pad (shouldn't happen often)
    r32 = r32[:16]
    return r32

def simulate_knockout(pairings):
    # Single elimination; no draws (penalties if needed)
    winners = []
    for a,b in pairings:
        ra = float(teams.loc[teams.Team==a,"Rating"].values[0])
        rb = float(teams.loc[teams.Team==b,"Rating"].values[0])
        pa = bt_win_prob(ra, rb)
        r = rng.random()
        winners.append(a if r < pa else b)
    return winners

def tournament_once():
    groups = seed_groups(teams)
    standings = simulate_group(groups)
    r32 = pick_knockout(standings)
    r16 = simulate_knockout(r32)
    qf  = simulate_knockout([(r16[i], r16[i+1]) for i in range(0, len(r16), 2)])
    sf  = simulate_knockout([(qf[i], qf[i+1]) for i in range(0, len(qf), 2)])
    fi  = simulate_knockout([(sf[0], sf[1])])
    champ = fi[0]
    return champ, standings, r32, r16, qf, sf, fi

# ---------------- Run simulations ----------------
counts = defaultdict(int)
sample_bracket = None
for _ in range(sims):
    champ, standings, r32, r16, qf, sf, fi = tournament_once()
    counts[champ] += 1
    if sample_bracket is None:
        sample_bracket = (standings, r32, r16, qf, sf, fi)

# ---------------- Output ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🏆 Championship Odds")
probs = pd.DataFrame([{"Team": k, "Win%": 100*counts[k]/sims} for k in sorted(counts.keys())])
probs = probs.sort_values("Win%", ascending=False).reset_index(drop=True)
st.dataframe(probs.head(20), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

if sample_bracket:
    standings, r32, r16, qf, sf, fi = sample_bracket
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧩 One Sample Tournament (for illustration)")
    # Groups
    st.markdown("**Group Stage (sample)**")
    cols = st.columns(4)
    for i,g in enumerate(GROUP_LABELS):
        with cols[i%4]:
            st.markdown(f"### Group {g}")
            box = ""
            for (team, s) in standings[g]:
                box += f"- {team} — {s['pts']} pts (GD {s['gd']}, GF {s['gf']})\n"
            st.text(box.strip())
    st.markdown("**Round of 32 (sample)**")
    st.write(", ".join([f"{a} vs {b}" for a,b in r32]))
    st.markdown("**Final (sample)**")
    st.write(f"{fi[0]} — Champion")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<p class="subtle">Note: Format approximates 2026 (12 groups of 4 → best 8 third-places). Bracketing is simplified.</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
