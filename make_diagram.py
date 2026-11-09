import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

fig, ax = plt.subplots(figsize=(11, 7)); ax.set_xlim(0, 11); ax.set_ylim(0, 7); ax.axis("off")
ax.add_patch(Rectangle((2.6, 0.3), 5.8, 6.4, fill=False, lw=2))
ax.text(5.5, 6.45, "Student Service Request SLA Monitoring System", ha="center", weight="bold")
ucs = ["Upload & validate dataset", "View exploratory analysis", "Predict SLA breach risk",
       "View risk / review queue", "Record final decision", "Export decision log", "Maintain model & deployment"]
ys = [5.7, 4.8, 3.9, 3.0, 2.1, 1.3, 0.8]
pos = {}
for u, y in zip(ucs, ys):
    ax.add_patch(Ellipse((5.5, y), 3.8, 0.7, fill=True, fc="#e8f0fe", ec="#1a4fa0")); ax.text(5.5, y, u, ha="center", va="center", fontsize=9)
    pos[u] = y
def actor(x, y, name):
    ax.plot(x, y + .35, "o", ms=14, mfc="white", mec="k"); ax.plot([x, x], [y + .2, y - .35], "k"); ax.plot([x - .3, x + .3], [y, y], "k")
    ax.plot([x, x - .25], [y - .35, y - .8], "k"); ax.plot([x, x + .25], [y - .35, y - .8], "k"); ax.text(x, y - 1.05, name, ha="center", fontsize=9)
actor(1.2, 4.2, "Student Services\nStaff"); actor(1.2, 1.6, "Office Head /\nSupervisor"); actor(9.8, 4.2, "System\nAdministrator")
for u in ucs[:6]:  ax.plot([1.5, 3.6], [4.2, pos[u]], color="gray", lw=.8)
for u in ucs[3:6]: ax.plot([1.5, 3.6], [1.6, pos[u]], color="gray", lw=.8)
ax.plot([9.5, 7.4], [4.2, pos[ucs[6]]], color="gray", lw=.8); ax.plot([9.5, 7.4], [4.2, pos[ucs[0]]], color="gray", lw=.8)
ax.text(5.5, 0.1, "Predictions support decisions; humans record all final decisions.", ha="center", fontsize=8, style="italic")
plt.savefig("use_case_diagram.png", dpi=150, bbox_inches="tight")
