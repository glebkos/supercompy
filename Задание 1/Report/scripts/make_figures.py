import argparse
import json
import math
from pathlib import Path
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Polygon, Rectangle
ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=Path, required=True)
args = parser.parse_args()
DATA = args.data.resolve()
TABLES, FIG = ROOT/'tables', ROOT/'figures'
for path in Path('/System/Library/Fonts/Supplemental').glob('Times New Roman*.ttf'):
    font_manager.fontManager.addfont(str(path))
font_manager.findfont('Times New Roman', fallback_to_default=False)
plt.rcParams.update({'font.family':'Times New Roman','font.size':13.5,
    'mathtext.fontset':'stix','axes.labelsize':14,'legend.fontsize':12,
    'xtick.labelsize':12.5,'ytick.labelsize':12.5,'axes.linewidth':.7,
    'lines.linewidth':1.5,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':240,
    'axes.spines.top':False,'axes.spines.right':False})
COLORS = ['#4878D0','#EE854A','#6ACC64','#D65F5F']
summary=json.loads((DATA/'measurements_summary.json').read_text())
raw=[json.loads(line) for line in (DATA/'measurements_raw.jsonl').read_text().splitlines()]
manufactured=json.loads((DATA/'manufactured.json').read_text())
def key(r): return r['mode'],r['nx'],r['ny'],r['processes'],r['threads']
assert len(summary)==37 and len(set(map(key,summary)))==37
assert len(raw)==193 and sum(r['phase']=='measured' for r in raw)==155
baseline={(r['nx'],r['ny']):r for r in summary if r['mode']=='seq'}
for row in summary:
    repeats=[r for r in raw if key(r)==key(row) and r['phase']=='measured']
    assert len(repeats)==row['repeats']==(7 if row['nx']<=80 else 3)
    assert all(r['converged'] and r['restarts']==0 and r['step_l2']<r['delta'] for r in repeats)
    assert row['solve_seconds']==statistics.median(r['solve_seconds'] for r in repeats)
    assert row['iterations_min']==row['iterations_max']
    base=baseline[row['nx'],row['ny']]
    assert row['speedup']==base['solve_seconds']/row['solve_seconds']
    row['speedup_low']=base['solve_min']/row['solve_max']
    row['speedup_high']=base['solve_max']/row['solve_min']

def save(fig,name):
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)
def panel(ax,label):
    ax.text(-.13,1.04,label,transform=ax.transAxes,fontweight='bold',fontsize=14)

ys = np.linspace(-5, 5, 1001)
width = np.sqrt(np.maximum(0, 9-(np.abs(ys)-2)**2))
outline = np.column_stack([np.r_[-width, width[::-1]], np.r_[ys, ys[::-1]]])
fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.0), layout='constrained')
for k, ax in enumerate(axes):
    ax.set(xlim=(-4.5, 4.5), ylim=(-6.5, 6.5), xlabel='$x$', ylabel='$y$')
    ax.set_aspect('equal')
    ax.set_xticks([-4, 0, 4]); ax.set_yticks([-6, -3, 0, 3, 6])
    ax.add_patch(Rectangle((-4,-6), 8, 12, fc='#F3F3F3', ec='0.3', lw=.8))
    panel(ax, 'ab'[k])
axes[0].add_patch(Polygon(outline, fc=COLORS[0], alpha=.25, ec='none'))
axes[0].add_patch(Polygon(outline, fill=False, ec='black', lw=1.3))
axes[0].text(0, 2, '$D$', ha='center', fontsize=16)
axes[0].text(3.2, 5.1, r'$\Pi$', ha='center', fontsize=15)
for cx in range(2):
    for cy in range(2):
        rank = 2*cx+cy
        axes[1].add_patch(Rectangle((-4+4*cx, -6+6*cy), 4, 6,
                                    fc=COLORS[rank], alpha=.20, ec='0.35', lw=.7))
        axes[1].text(-2+4*cx, -3+6*cy, f'P{rank}', ha='center', va='center')
axes[1].add_patch(Polygon(outline, fill=False, ec='black', lw=1.3))
save(fig, 'geometry')


large=np.loadtxt(DATA/'solution_800_1200.csv.gz',delimiter=',',skiprows=1)
assert large.shape==(801*1201,4)
u=large[:,2].reshape(801,1201)
mask=large[:,3].reshape(801,1201).astype(bool)
field=np.ma.masked_where(~mask.T,u.T)
fig,axes=plt.subplots(1,2,figsize=(6.5,4.2),layout='constrained',gridspec_kw={'width_ratios':[1,1.25]})
norm=TwoSlopeNorm(vmin=float(field.min()),vcenter=0,vmax=float(field.max()))
im=axes[0].imshow(field,origin='lower',extent=(-4.005,4.005,-6.005,6.005),cmap='RdBu_r',norm=norm,interpolation='nearest',rasterized=True)
axes[0].add_patch(Polygon(outline,fill=False,ec='0.2',lw=.65))
axes[0].set(xlim=(-3.25,3.25),ylim=(-5.25,5.25),xlabel='$x$',ylabel='$y$')
cbar=fig.colorbar(im,ax=axes[0],shrink=.83,pad=.04);cbar.ax.set_title('$v$',pad=8)
y=large[:,1].reshape(801,1201)[400]
cut=np.abs(y)<=5
axes[1].plot(y[cut],u[400,cut],color=COLORS[0],label='Computed $v$')
axes[1].plot(y[cut],np.sin(y[cut]),color='0.4',ls='--',label='Extension $g$')
axes[1].set(xlabel='$y$ at $x=0$',ylabel='$v(0,y)$',xlim=(-5,5))
axes[1].grid(alpha=.2);axes[1].legend(loc='upper right',frameon=False,fontsize=11.5)
for ax,label in zip(axes,'ab'): panel(ax,label)
save(fig,'solution')

fig,ax=plt.subplots(figsize=(5.6,3.2),layout='constrained')
h=np.array([12/r['nx'] for r in manufactured]);errors=np.array([r['exact_max_error'] for r in manufactured])
ax.loglog(h,errors,'o-',color=COLORS[0],label='Computed error')
ax.loglog(h,errors[-1]*(h/h[-1])**2,'--',color='0.45',label='$C h^2$ reference')
ax.set(xlabel='Maximum grid spacing $h$',ylabel='Maximum error')
ax.grid(which='both',alpha=.2);ax.legend(frameon=False)
save(fig,'convergence')

labels={'omp':'OpenMP','mpi':'MPI','hybrid':'MPI+OpenMP'}
def draw(ax,rows,color,marker,label,style='-',fill=True):
    x=np.array([r['processes']*r['threads'] for r in rows])
    y=np.array([r['speedup'] for r in rows])
    err=np.array([[r['speedup']-r['speedup_low'] for r in rows],[r['speedup_high']-r['speedup'] for r in rows]])
    ax.errorbar(x,y,yerr=err,marker=marker,color=color,capsize=3,label=label,ls=style,mfc=color if fill else 'white')
fig,ax=plt.subplots(figsize=(6.0,3.4),layout='constrained')
for mode,color,marker in zip(('omp','mpi','hybrid'),COLORS,('o','s','^')):
    draw(ax,[r for r in summary if r['nx']==80 and r['mode']==mode],color,marker,labels[mode])
ax.set(xlabel='Processes x threads',ylabel='Speedup',xscale='log',ylim=(0,2.7))
ax.set_xticks([1,2,4,8,16],['1','2','4','8','16'])
ax.minorticks_off()
ax.axhline(1,color='0.4',ls=':',lw=.9);ax.grid(alpha=.2);ax.legend(frameon=False,loc='upper left')
save(fig,'speedup_small')

fig,axes=plt.subplots(3,1,figsize=(6.5,7.6))
fig.subplots_adjust(left=.13,right=.98,bottom=.075,top=.96,hspace=.56)
for ax,mode,color,letter in zip(axes,('omp','mpi','hybrid'),COLORS,'abc'):
    for nx,marker,style in ((400,'o','-'),(800,'s','--')):
        rows=[r for r in summary if r['nx']==nx and r['mode']==mode]
        lab='%d x %d' % (nx,rows[0]['ny'])
        if mode=='hybrid': lab+=', P=%d' % rows[0]['processes']
        x=np.array([r['threads'] if mode!='mpi' else r['processes'] for r in rows])
        y=np.array([r['speedup'] for r in rows])
        err=np.array([[r['speedup']-r['speedup_low'] for r in rows],[r['speedup_high']-r['speedup'] for r in rows]])
        ax.errorbar(x,y,yerr=err,marker=marker,color=color,capsize=3,ls=style,label=lab,mfc=color if nx==400 else 'white')
    ticks=[1,2,4,8] if mode=='hybrid' else [2,4,8,16,32]
    ax.set(xscale='log',ylabel='Speedup',xlabel='MPI processes' if mode=='mpi' else ('Threads per process' if mode=='hybrid' else 'OpenMP threads'))
    ax.set_xticks(ticks,[str(v) for v in ticks]);ax.set_ylim(bottom=0)
    ax.minorticks_off()
    ax.axhline(1,color='0.45',ls=':',lw=.8);ax.grid(alpha=.2)
    ax.legend(loc='upper left',frameon=False,fontsize=11.5)
    ax.set_title(labels[mode],loc='left',pad=8,fontsize=14)
    ax.text(-.12,1.06,letter,transform=ax.transAxes,fontweight='bold',fontsize=14)
save(fig,'speedup_large')

def sci(v):
    exponent=int(math.floor(math.log10(abs(v)))) if v else 0
    return f'${v/10**exponent:.2f}\\cdot10^{{{exponent}}}$'
def timefmt(v): return f'{v:.6f}' if v<1 else f'{v:.3f}'
table_index={}
def save_rows(name,lines,rows):
    (TABLES/(name+'.tex')).write_text('\n'.join(line+r' \\' for line in lines)+'\n')
    table_index[name]=[key(r) for r in rows]
seq=[r for r in summary if r['mode']=='seq']
save_rows('sequential_rows',[f"{r['nx']}\\,$\\times$\\,{r['ny']} & {r['iterations_min']} & {timefmt(r['solve_seconds'])} & [{timefmt(r['solve_min'])}; {timefmt(r['solve_max'])}] & {sci(r['residual_max'])}" for r in seq],seq)
for mode in ('omp','mpi','hybrid'):
    rows=[r for r in summary if r['mode']==mode and r['nx']==80]
    lines=[]
    for r in rows:
        resource=f"{r['processes']} & {r['threads']}" if mode=='hybrid' else str(r['threads'] if mode=='omp' else r['processes'])
        lines.append(f"{resource} & {r['iterations_min']} & {r['solve_seconds']*1000:.3f} & [{r['solve_min']*1000:.3f}; {r['solve_max']*1000:.3f}] & {r['speedup']:.3f}")
    save_rows(mode+'_rows',lines,rows)
for mode,ncols in (('omp',5),('mpi',6),('hybrid',7)):
    lines=[];used=[]
    for nx,ny in ((400,600),(800,1200)):
        lines.append(r'\multicolumn{'+str(ncols)+r'}{c}{'+f'Сетка ${nx}\\times {ny}$'+r'}')
        rows=[r for r in summary if r['mode']==mode and r['nx']==nx]
        for r in rows:
            resource=f"{r['processes']} & {r['threads']}" if mode=='hybrid' else str(r['threads'] if mode=='omp' else r['processes'])
            node='' if mode=='omp' else f" & {r['nodes']}"
            lines.append(f"{resource} & {r['iterations_min']}{node} & {r['solve_seconds']:.3f} & [{r['solve_min']:.3f}; {r['solve_max']:.3f}] & {r['speedup']:.3f}")
            used.append(r)
    save_rows('large_'+mode+'_rows',lines,used)
lines=[]
for k,r in enumerate(manufactured):
    rate='н/д' if k==0 else f"{math.log2(manufactured[k-1]['exact_max_error']/r['exact_max_error']):.3f}"
    lines.append(f"{r['nx']}\\,$\\times$\\,{r['ny']} & {12/r['nx']:.3f} & {sci(r['exact_max_error'])} & {rate}")
(TABLES/'accuracy_rows.tex').write_text('\n'.join(line+r' \\' for line in lines)+'\n')
flattened=[tuple(k) for group in table_index.values() for k in group]
assert len(flattened)==37 and set(flattened)==set(map(key,summary))
print('37 configurations verified; five PDF figures and eight table bodies generated')
