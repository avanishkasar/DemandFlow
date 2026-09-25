"""
Generate Fig 9a, 9b, 9c — print-ready, large fonts, high contrast
"""
import numpy as np
import matplotlib.pyplot as plt
import os

OUT = 'graphs'
os.makedirs(OUT, exist_ok=True)
DPI = 300

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         12,
    'axes.titlesize':    15,
    'axes.titleweight':  'bold',
    'axes.labelsize':    13,
    'axes.labelweight':  'bold',
    'xtick.labelsize':   11,
    'ytick.labelsize':   11,
    'legend.fontsize':   11,
    'legend.framealpha': 0.92,
    'savefig.dpi':       DPI,
    'savefig.bbox':      'tight',
    'savefig.facecolor': 'white',
    'axes.facecolor':    '#FAFAFA',
    'axes.grid':         True,
    'grid.color':        '#dddddd',
    'grid.linewidth':    0.8,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'lines.linewidth':   2.5,
})

C = {'blue':'#1565C0','green':'#2E7D32','red':'#C62828','gold':'#F57F17',
     'orange':'#E65100','grey':'#546E7A','navy':'#0D1B2A','purple':'#6A1B9A'}

np.random.seed(42)
base_price  = 360.72
prices      = np.linspace(base_price*0.7, base_price*1.3, 40)
base_demand = 1800
demand      = base_demand * (1 - 0.8*((prices - base_price*0.7)/(base_price*0.6))**1.3)
demand      = np.maximum(demand + np.random.normal(0, 28, len(demand)), 50)
revenue     = prices * demand
opt_idx     = np.argmax(revenue)
opt_price   = prices[opt_idx]
opt_rev     = revenue[opt_idx]
opt_dem     = demand[opt_idx]

# ── FIG 9a — Demand Curve ────────────────────────────────────────────────────
print("Fig 9a ...")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(prices, demand, color=C['blue'], lw=2.8, marker='o', ms=4.5,
        label='Predicted Weekly Demand')
ax.fill_between(prices, demand, alpha=0.12, color=C['blue'])

ax.axvline(opt_price, color=C['red'], ls='--', lw=2, alpha=0.8)
ax.plot(opt_price, opt_dem, 'D', color=C['red'], ms=12, zorder=10,
        label=f'At optimal price (INR {opt_price:.0f}): {opt_dem:.0f} units')
ax.annotate(f'INR {opt_price:.0f}\n(Revenue-maximising\nprice point)',
            xy=(opt_price, opt_dem),
            xytext=(opt_price + 22, opt_dem + 150),
            fontsize=11, fontweight='bold', color=C['red'],
            arrowprops=dict(arrowstyle='->', color=C['red'], lw=2),
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor=C['red'], alpha=0.95))

mid = len(prices)//2
ax.annotate('High elasticity\n(demand drops steeply)',
            xy=(prices[mid+4], demand[mid+4]),
            xytext=(prices[mid+6], demand[mid+4]+300),
            fontsize=10, fontweight='bold', color=C['grey'],
            arrowprops=dict(arrowstyle='->', color=C['grey'], lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=C['grey'], alpha=0.9))
ax.annotate('Lower elasticity\n(budget buyers)',
            xy=(prices[5], demand[5]),
            xytext=(prices[5]+18, demand[5]-420),
            fontsize=10, fontweight='bold', color=C['grey'],
            arrowprops=dict(arrowstyle='->', color=C['grey'], lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=C['grey'], alpha=0.9))

ax.set_xlabel('Launch Price (INR)')
ax.set_ylabel('Predicted Weekly Demand (Units)')
ax.set_title('Fig. 9a: Price–Demand Relationship\nPredicted demand decreases as launch price increases')
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig(f'{OUT}/fig09a_demand_curve.png')
plt.close()

# ── FIG 9b — Revenue Curve ───────────────────────────────────────────────────
print("Fig 9b ...")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(prices, revenue/1000, color=C['green'], lw=2.8, ls='--', marker='s', ms=4.5,
        label='Predicted Weekly Revenue')
ax.fill_between(prices, revenue/1000, alpha=0.12, color=C['green'])

ax.axvline(opt_price, color=C['gold'], ls='--', lw=2.2, alpha=0.9)
ax.plot(opt_price, opt_rev/1000, '*', color=C['gold'], ms=24,
        markeredgecolor='black', markeredgewidth=2, zorder=10,
        label=f'Peak Revenue: INR {opt_rev/1000:.1f}K at INR {opt_price:.0f}')
ax.axhline(opt_rev/1000, color=C['gold'], ls=':', lw=1.5, alpha=0.6)

ax.axvspan(prices[0],   opt_price, alpha=0.06, color=C['orange'])
ax.axvspan(opt_price, prices[-1], alpha=0.06, color=C['red'])

ax.text(prices[7],  opt_rev/1000*0.80, 'Under-priced zone:\nleft revenue on the table',
        fontsize=10, fontweight='bold', color=C['orange'],
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                  edgecolor=C['orange'], alpha=0.92))
ax.text(prices[-14], opt_rev/1000*0.72, 'Over-priced zone:\ndemand drops steeply',
        fontsize=10, fontweight='bold', color=C['red'],
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                  edgecolor=C['red'], alpha=0.92))

ax.set_xlabel('Launch Price (INR)')
ax.set_ylabel('Predicted Weekly Revenue (INR Thousands)')
ax.set_title('Fig. 9b: Price–Revenue Relationship\nRevenue peaks at the optimal price point (★)')
ax.legend(loc='lower center', fontsize=11)
plt.tight_layout()
plt.savefig(f'{OUT}/fig09b_revenue_curve.png')
plt.close()

# ── FIG 9c — Recommendation Summary Panel ────────────────────────────────────
print("Fig 9c ...")
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle('Fig. 9c: Price Optimisation – Recommendation Summary',
             fontweight='bold', fontsize=16, y=1.02)

# Panel 1 — KPI boxes
ax = axes[0]
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
kpis = [
    ('Optimal Launch Price',    f'INR {opt_price:.0f}',    C['green']),
    ('Expected Weekly Units',   f'{opt_dem:.0f} units',    C['blue']),
    ('Expected Weekly Revenue', f'INR {opt_rev/1000:.1f}K',C['gold']),
    ('vs. Current Price',       f'{((opt_price/base_price)-1)*100:+.1f}%', C['purple']),
]
for i, (label, val, col) in enumerate(kpis):
    y = 0.86 - i*0.23
    ax.add_patch(plt.Rectangle((0.05, y-0.09), 0.9, 0.19,
                                facecolor=col, alpha=0.14,
                                edgecolor=col, linewidth=2.2,
                                transform=ax.transAxes))
    ax.text(0.5, y+0.04, val, ha='center', va='center',
            fontsize=15, fontweight='bold', color=col, transform=ax.transAxes)
    ax.text(0.5, y-0.04, label, ha='center', va='center',
            fontsize=10, color=C['navy'], transform=ax.transAxes)
ax.set_title('Key Metrics', fontsize=13)

# Panel 2 — Demand at 7 price points
ax = axes[1]
test_prices = np.linspace(prices[0], prices[-1], 7)
test_demand = base_demand*(1 - 0.8*((test_prices - base_price*0.7)/(base_price*0.6))**1.3)
test_demand = np.maximum(test_demand, 50)
bar_colors  = [C['green'] if abs(p-opt_price)<18 else
               (C['orange'] if p<opt_price else C['red']) for p in test_prices]
bars = ax.bar([f'INR\n{p:.0f}' for p in test_prices], test_demand,
              color=bar_colors, edgecolor='black', lw=0.9, alpha=0.85, width=0.65)
for bar, val in zip(bars, test_demand):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+18,
            f'{val:.0f}', ha='center', fontsize=10.5, fontweight='bold')
ax.set_ylabel('Predicted Demand (Units)')
ax.set_xlabel('Price Point')
ax.set_title('Demand at 7 Price Points\nGreen = Optimal range')

# Panel 3 — Revenue at 7 price points
ax = axes[2]
test_revenue = test_prices * test_demand / 1000
rev_colors   = [C['green'] if abs(p-opt_price)<18 else
                (C['orange'] if p<opt_price else C['red']) for p in test_prices]
bars = ax.bar([f'INR\n{p:.0f}' for p in test_prices], test_revenue,
              color=rev_colors, edgecolor='black', lw=0.9, alpha=0.85, width=0.65)
for bar, val in zip(bars, test_revenue):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
            f'{val:.1f}K', ha='center', fontsize=10.5, fontweight='bold')
ax.set_ylabel('Predicted Revenue (INR Thousands)')
ax.set_xlabel('Price Point')
ax.set_title('Revenue at 7 Price Points\nGreen = Optimal range')

plt.tight_layout()
plt.savefig(f'{OUT}/fig09c_price_recommendation.png')
plt.close()

print("\nDone! fig09a/b/c saved.")
