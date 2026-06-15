import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Polygon, FancyBboxPatch

fig, ax = plt.subplots(figsize=(24, 10))
ax.set_xlim(0, 24)
ax.set_ylim(0, 10)
ax.axis('off')
fig.patch.set_facecolor('#FEFEFE')
ax.set_facecolor('#FEFEFE')

PINK   = '#E05C4B'
BLUE   = '#4A8BD4'
GREEN  = '#43A047'
GRAY   = '#8E8E8E'
ORANGE = '#FB8C00'
TEAL   = '#26A69A'
PURPLE = '#7B1FA2'
RED    = '#D32F2F'


def draw_circle(x, y, r=0.5, color='blue', label='', fs=8.5):
    ax.add_patch(Circle((x, y), r, color=color, zorder=3))
    lines = label.split('\n')
    n = len(lines)
    for i, ln in enumerate(lines):
        dy = (n - 1 - 2 * i) * 0.14
        ax.text(x, y + dy, ln, ha='center', va='center', fontsize=fs,
                fontweight='bold', color='white', zorder=4)


def draw_diamond(x, y, w=0.8, h=0.5, color=TEAL, label='', fs=7.5):
    pts = [[x, y + h], [x + w, y], [x, y - h], [x - w, y]]
    ax.add_patch(Polygon(pts, color=color, zorder=3))
    lines = label.split('\n')
    n = len(lines)
    for i, ln in enumerate(lines):
        dy = (n - 1 - 2 * i) * 0.12
        ax.text(x, y + dy, ln, ha='center', va='center', fontsize=fs,
                fontweight='bold', color='white', zorder=4)


def arr(x1, y1, x2, y2, label='', color='#444', fs=7, rad=0, lo=(0, 0.18)):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5,
                                connectionstyle=f'arc3,rad={rad}'))
    if label:
        ax.text((x1 + x2) / 2 + lo[0], (y1 + y2) / 2 + lo[1],
                label, ha='center', va='bottom', fontsize=fs, color=color, style='italic')


# ── Section labels & dividers ────────────────────────────────────────────────
ax.text(3.2,  9.65, 'Query Analysis',           ha='center', fontsize=11, color=PINK,   fontweight='bold', fontstyle='italic')
ax.text(8.5,  9.65, 'Retrieval + Hybrid Search', ha='center', fontsize=11, color=BLUE,   fontweight='bold', fontstyle='italic')
ax.text(18.5, 9.65, 'Generate + Self-Reflection', ha='center', fontsize=11, color=ORANGE, fontweight='bold', fontstyle='italic')
ax.axvline(5.5,  ymin=0.03, ymax=0.94, color='#DDD', lw=1, ls='--', zorder=0)
ax.axvline(12.5, ymin=0.03, ymax=0.94, color='#DDD', lw=1, ls='--', zorder=0)

# ── Question ─────────────────────────────────────────────────────────────────
ax.text(0.1, 5, 'Question', va='center', fontsize=10.5, fontweight='bold', color='#333')
arr(1.3, 5, 1.8, 5, color='#555')

# ── Router ───────────────────────────────────────────────────────────────────
draw_circle(2.35, 5, r=0.5, color=PINK, label='Query\nRouter')
arr(2.85, 5, 3.35, 5, color='#555')
draw_diamond(4.1, 5, w=0.72, h=0.46, color=PINK, label='Route?', fs=8)

# ── Vectorstore branch (y = 8) ───────────────────────────────────────────────
arr(4.1, 5.46, 4.1, 7.48, color=BLUE)
ax.text(4.35, 6.47, '[vectorstore]', fontsize=7, color=BLUE, style='italic')
draw_circle(4.1, 8, r=0.48, color=BLUE, label='Vectorstore', fs=8)
arr(4.58, 8, 5.9, 8, color=BLUE)
draw_circle(6.5, 8, r=0.55, color=BLUE, label='Retrieve', fs=9)
arr(7.05, 8, 7.95, 8, color=BLUE)
draw_diamond(8.75, 8, w=0.78, h=0.5, color=TEAL, label='Grade\nDocs', fs=7.5)

# all relevant → generate
arr(9.53, 8, 13.4, 8, label='all relevant', color=GREEN, fs=7)

# any irrelevant → Web Search
arr(8.75, 7.5, 8.75, 5.55, color=ORANGE)
ax.text(9.1, 6.5, 'any irrelevant', fontsize=6.5, color=ORANGE, style='italic')
draw_circle(8.75, 5, r=0.52, color=GREEN, label='Web\nSearch', fs=8.5)

# Web Search → Generate (merging docs)
ax.annotate('', xy=(13.4, 7.5), xytext=(9.27, 5.0),
            arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.5,
                            connectionstyle='arc3,rad=-0.22'))
ax.text(11.9, 5.7, 'merge + vector docs', fontsize=7, color=ORANGE, style='italic', ha='center')

# ── LLM Fallback branch (y = 2) ──────────────────────────────────────────────
arr(4.1, 4.54, 4.1, 2.55, color=GRAY)
ax.text(4.38, 3.5, '[llm_fallback]', fontsize=7, color=GRAY, style='italic')
draw_circle(4.1, 2, r=0.52, color=GRAY, label='LLM\nFallback', fs=8.5)
arr(4.62, 2, 7.2, 2, color=GRAY)
ax.text(7.3, 2, 'Answer w/ LLM', va='center', fontsize=9, fontweight='bold', color=GRAY)

# ── Generate node ─────────────────────────────────────────────────────────────
draw_circle(14.0, 8, r=0.58, color=ORANGE, label='Generate', fs=9.5)

# Mode info box
ax.add_patch(FancyBboxPatch((13.1, 6.55), 1.85, 0.9, boxstyle='round,pad=0.1',
                             facecolor='#FFF8E1', edgecolor='#FFB300', lw=1.2, zorder=2))
ax.text(14.0, 7.1,  'ask     →  factual',   ha='center', fontsize=6.5, color='#E65100', fontweight='bold')
ax.text(14.0, 6.88, 'idea    →  creative',  ha='center', fontsize=6.5, color=PURPLE,   fontweight='bold')
ax.text(14.0, 6.66, 'explore →  research',  ha='center', fontsize=6.5, color='#1B5E20', fontweight='bold')

# HTTP tool badge
ax.add_patch(FancyBboxPatch((13.1, 6.0), 1.85, 0.45, boxstyle='round,pad=0.08',
                             facecolor='#EDE7F6', edgecolor=PURPLE, lw=1.0, zorder=2))
ax.text(14.0, 6.225, 'HTTP Tool  (all methods)', ha='center', va='center',
        fontsize=7, color=PURPLE, fontweight='bold')

# ── idea → END directly (dashed) ─────────────────────────────────────────────
ax.annotate('', xy=(14.0, 9.7), xytext=(14.0, 8.58),
            arrowprops=dict(arrowstyle='->', color=PURPLE, lw=1.3,
                            linestyle='dashed', connectionstyle='arc3,rad=0'))
ax.text(14.45, 9.15, 'idea: no grading', fontsize=6.5, color=PURPLE, style='italic')
ax.text(14.0, 9.9, 'END (idea)', ha='center', fontsize=9, fontweight='bold', color=PURPLE)

# ── ask / explore → Grade Generation ─────────────────────────────────────────
arr(14.58, 8, 16.5, 8, label='ask / explore', color='#555', fs=7)
draw_diamond(17.3, 8, w=0.82, h=0.52, color=TEAL, label='Grade\nGeneration', fs=7.5)

# grounded? → Answers Question?
arr(18.12, 8, 19.3, 8, label='grounded?', color='#555', fs=6.5, lo=(0, 0.22))
draw_diamond(20.1, 8, w=0.8, h=0.5, color=TEAL, label='Answers\nQuestion?', fs=7)

# yes → Answer END
arr(20.9, 8, 21.6, 8, color=GREEN)
ax.text(20.9, 8.28, 'yes', fontsize=7, color=GREEN, style='italic')
ax.text(21.65, 8, 'Answer', va='center', fontsize=10, fontweight='bold', color=GREEN)

# not useful → Web Search (big back-loop)
ax.annotate('', xy=(8.75, 4.48), xytext=(20.1, 7.5),
            arrowprops=dict(arrowstyle='->', color=RED, lw=1.3,
                            connectionstyle='arc3,rad=0.38'))
ax.text(15.0, 2.85, 'no (not useful)  →  web search', fontsize=7.5, color=RED,
        style='italic', ha='center')
ax.text(19.7, 7.4, 'no', fontsize=7, color=RED, style='italic', ha='right')

# hallucination → retry generate
ax.annotate('', xy=(14.58, 8.1), xytext=(17.3, 8.52),
            arrowprops=dict(arrowstyle='->', color=RED, lw=1.3,
                            connectionstyle='arc3,rad=-0.4'))
ax.text(15.9, 9.38, 'hallucination  →  retry', fontsize=7, color=RED,
        style='italic', ha='center')

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(12, 0.4, 'Acela AI Agent — Current Multi-Mode Graph Architecture',
        ha='center', fontsize=12, fontweight='bold', color='#2C2C2C')

plt.tight_layout(pad=0.2)
plt.savefig('docs/architecture.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: docs/architecture.png")
