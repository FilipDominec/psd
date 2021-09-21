matplotlib.rc('font', size=12, family='serif')

for x,y in zip(xs,ys): y *= x**'0 

# compute smooth weighted average of all selected curves
AVERAGE_POINTS = 1000      # resulting curve resolution
XLOG, YLOG = 1, 1   # affects the averaging weights if data are to be shown in log plots
if XLOG: xs = [np.log10(x) for x in xs]
if YLOG: ys = [np.log10(y) for y in ys]
minx, maxx = np.min([np.min(x) for x in xs]), np.max([np.max(x) for x in xs])
wxs, wys, wws = np.linspace(minx,maxx, AVERAGE_POINTS), np.zeros(AVERAGE_POINTS), np.zeros(AVERAGE_POINTS)
for (x,y) in zip(xs,ys):    
    centerx,extx = np.min(x)/2+np.max(x)/2, np.max(x)/2-np.min(x)/2
    weight = np.exp(-((wxs-centerx)*1.3/extx)**4)
    weight[np.logical_or(wxs<np.min(x),wxs>np.max(x))] = 0
    if XLOG: weight *= np.exp(((wxs-centerx)*1/extx)) # optional: more weight on right side of function
    wys += np.interp(wxs,x,y) * weight
    wws += weight
if XLOG: xs = [10**(x) for x in xs]
if YLOG: ys = [10**(y) for y in ys]
resulting_x, resulting_y = 10**wxs if XLOG else wxs, 10**(wys/wws) if YLOG else (wys/wws)
ax.plot(6.28/resulting_x, resulting_y, color='k', lw=2.5)
print(resulting_x, resulting_y)

for          x,  y,  n,              param,  label,  xlabel,  ylabel,  color in \
         zip(xs, ys, range(len(xs)), params, labels, xlabels, ylabels, colors):
    # x, y = x[~np.isnan(y)], y[~np.isnan(y)]        ## filter-out NaN points
    # convol = 2**-np.linspace(-2,2,25)**2; y = np.convolve(y,convol/np.sum(convol), mode='same') ## simple smoothing

    ax.plot(6.28/x, y, label="%s" % (label), color=color)
    #ax.plot(x, y, label="%s" % (label.split('.dat')[0]), color=colors[c%10], ls=['-','--'][int(c/10)]) 
ax.set_xlabel(xlabelsdedup)
ax.set_ylabel(ylabelsdedup)

plot_title = sharedlabels[-4:] ## last few labels that are shared among all curves make a perfect title
#plot_title = sharedlabels[sharedlabels.index('LastCroppedLabel')+1:] ## optionally, use all labels after the chosen one 

#ax.set_xlim(xmin=0, xmax=1)
#ax.set_ylim(ymin=2.6, ymax=2.7)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title(' '.join(plot_title)) 
ax.legend(loc='best', prop={'size':10})

#np.savetxt('output.dat', np.vstack([x,ys[0],ys[1]]).T, fmt="%.8g")
#tosave.append('_'.join(plot_title)+'.png') ## whole graph will be saved as PNG
#tosave.append('_'.join(plot_title)+'.pdf') ## whole graph will be saved as PDF
