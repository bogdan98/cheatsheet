# choose whatever font we want for the plots
matplotlib.rcParams.update({'font.size': 15})

1. Correlation map

# Compute the correlation matrix
corr = data.iloc[:, 1:].corr() #iloc on chosen locations of data

# Generate a mask for the upper triangle (optional)
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(8, 8))

# Generate a custom diverging colormap (optional)
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

# saving for the report (pdf works better with latex)
plt.savefig('name.png or name.pdf', bbox_inches='tight')


2. Joint plots (2D scatter + prob distribution of that scatter on x and y axes)

#x, y are pd.Series
sns.jointplot(x=x_col_name, y=y_col_name, data=data, kind="reg/hist/etc", marker = './+/etc., for scatterplot')


3. Multiple subplots

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 8))
plt.subplots_adjust(right=1.3)

sns.distplot(data.col1, ax=axes[0,0], label = 'label') # sns distplot
sns.distplot(data.col2, ax=axes[0,1])
axes[0,2].plot(x, y) # normal plot
# etc.

axes[0,0].set_ylabel(None)
axes[0,1].set_ylabel("Density")
axes[0,1].set_yticks(...)
axes[0,0].set_ylim([..., ...])
etc.

axes[0,0].legend() # legend with labels
# etc.


4. Subplots sharing an x-axis

fig = plt.figure()
ax1 = fig.add_subplot(111)

ln1 = ax1.plot(x1, y1, '.', markersize = 3, color = 'green', label = "label1")
ax1.set_yticks([0, 250, 500, 750, 1000, 1250])
ax1.set_ylim([0, 11.5])
ax1.set_ylabel('y1')
ax1.set_xlabel('x1')
ax1.yaxis.label.set_color('green')

# subplot sharing x axis
ax2 = ax1.twinx()

ln1 = ax2.plot(x2, y2, '.', markersize = 3, color = 'dodgerblue', label = "label2")
ax2.set_ylabel('y2')
ax2.set_xlim([70, 131])
ax2.set_ylim([1.8, 27])
ax2.yaxis.label.set_color('dodgerblue')

# set colors of ticks (optional), for a plot would be plt.yticks(color = 'color')
for t in ax1.yaxis.get_ticklabels(): 
    t.set_color('dodgerblue')
    
for t in ax2.yaxis.get_ticklabels(): 
    t.set_color('green')

# drawing arrows on a plot, works with a plot as well as subplots
ax1.annotate("", xy=(131, 3.9), xytext=(126, 3.9), arrowprops=dict(arrowstyle="->", lw = 3, color = 'dodgerblue'))
ax1.annotate("", xy=(70, 21.3), xytext=(75, 21.3), arrowprops=dict(arrowstyle="->", lw = 3, color = 'green'))

# vertical lines
ax1.vlines(92, 2.5, 14.2, linestyles = '--', linewidth = 0.5)
ax1.vlines(117, 2.5, 14.5, linestyles = '--', linewidth = 0.5)

# text annotation
ax1.text(76, 12, "some text")

5. Violin plot

sns.violinplot(x="day", y="some number", hue="nested grouping by two categorical variables",
                    data=dataframe)
                    
6. Bar plot

plt.bar(labels, means1, width1, yerr=std1, label='1')
plt.bar(labels, means2, width2, yerr=std2, bottom=means1(or zero, if no stacking),
       label='2')