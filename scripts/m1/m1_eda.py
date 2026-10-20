"""Descriptive charts only: no longitudinal depreciation or causal claims."""
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import json

COLORS=['#1768AC','#18A999','#E08E45','#755BB4','#DA627D','#627D98']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.titlesize':13,'figure.dpi':130,'savefig.dpi':180,'axes.grid':True,'grid.alpha':.15})
USD=FuncFormatter(lambda x,p:f'${x/1000:,.0f}k')

def create_eda(ev,root):
    root=Path(root); figdir=root/'figures'; figdir.mkdir(exist_ok=True)
    results=[]
    def save(fig,name,title,en,zh,table=None):
        fig.tight_layout(); fig.savefig(figdir/name,bbox_inches='tight',facecolor='white'); plt.close(fig)
        results.append(dict(file=name,title=title,takeaway_en=en,takeaway_zh=zh))
        if table is not None: table.to_csv(root/'data/diagnostics'/name.replace('.png','.csv'),index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4.3))
    axes[0].hist(ev.price,bins=60,color=COLORS[0],edgecolor='white',linewidth=.3); axes[0].set(title='Recorded asking price',xlabel='Asking price (USD)',ylabel='Listings'); axes[0].xaxis.set_major_formatter(USD)
    axes[1].hist(ev.log_price,bins=50,color=COLORS[1],edgecolor='white',linewidth=.3); axes[1].set(title='Natural log of asking price',xlabel='ln(asking price in USD)',ylabel='Listings')
    save(fig,'01_price_and_log_price.png','Price before and after taking the log',f'Raw-price skewness is {ev.price.skew():.2f}; log-price skewness is {ev.log_price.skew():.2f}. Log transformation reduces the high-price tail, but does not itself prove better predictive accuracy. All {len(ev):,} listings, including the project-designated error, are shown.',f'原价偏度为 {ev.price.skew():.2f}，取对数后为 {ev.log_price.skew():.2f}。高价尾部缩短，但是否提高预测精度仍需模型验证。图中保留全部 {len(ev):,} 条记录。')
    top=ev.model_line.value_counts().head(6).index.tolist()
    age=ev[ev.model_line.isin(top)].groupby(['model_line','age']).price.agg(median_price='median',listings='size').reset_index()
    fig,ax=plt.subplots(figsize=(11,5.2))
    for color,line in zip(COLORS,top):
        g=age[(age.model_line==line)&(age.listings>=5)].sort_values('age')
        ax.plot(g.age,g.median_price,'o-',label=line,color=color,linewidth=1.8,markersize=4)
    ax.set(title='Median asking price by age | six largest model families',xlabel='Approximate age: 2023 − model year',ylabel='Median asking price (USD)'); ax.yaxis.set_major_formatter(USD); ax.legend(loc='upper right',fontsize=8)
    ax.text(0,-.19,'Points require at least 5 listings. Cross-sectional snapshot; trim, mileage and condition are not held constant.',transform=ax.transAxes,fontsize=8,color='#52616B')
    save(fig,'02_price_by_age_top6.png','Median price by age, top six model families','Prices vary substantially by model family and age. These are comparisons of different cars in April 2023, not measurements of how one car loses value each year. Sparse points with fewer than five listings are omitted.', '不同车型、车龄对应的价格差别较大。这是同一时点不同车辆的对比，不能直接当成一辆车每年的贬值率。少于5辆的点不画。',age)
    topbrands=ev.manufacturer.value_counts().head(6).index.tolist()
    fig,ax=plt.subplots(figsize=(11,5.5))
    other=ev[~ev.manufacturer.isin(topbrands)]
    ax.scatter(other.mileage,other.price,s=9,alpha=.18,color='#B0BAC5',label='Other brands',rasterized=True)
    for color,brand in zip(COLORS,topbrands):
        g=ev[ev.manufacturer==brand]
        ax.scatter(g.mileage,g.price,s=10,alpha=.3,color=color,label=brand,rasterized=True)
    ax.set(title='Asking price versus mileage',xlabel='Mileage (miles)',ylabel='Asking price (USD)'); ax.yaxis.set_major_formatter(USD); ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x/1000:,.0f}k')); ax.legend(fontsize=8,ncol=2)
    rho=ev[['price','mileage']].corr(method='spearman').iloc[0,1]
    save(fig,'03_price_vs_mileage.png','Price versus mileage, coloured by brand',f'The pooled price–mileage rank correlation is {rho:.2f}. Brand and model differences remain visible, so this pooled association should not be interpreted as a mileage-only price effect.',f'价格与里程的整体秩相关系数为 {rho:.2f}。品牌和车型差异也明显，不能把整体关系理解为仅由里程造成的价格变化。')
    fig,axes=plt.subplots(2,3,figsize=(12,7.1),sharey=True)
    accident=[]
    for ax,line in zip(axes.ravel(),top):
        g=ev[ev.model_line==line]
        data=[g.loc[g.accidents_or_damage==k,'log_price'].to_numpy() for k in [0,1]]
        b=ax.boxplot(data,tick_labels=['No accident\nrecorded','Accident/\ndamage'],patch_artist=True,showfliers=False)
        for patch,color in zip(b['boxes'],[COLORS[0],COLORS[4]]): patch.set_facecolor(color); patch.set_alpha(.6)
        ax.set_title(line,fontsize=11); ax.set_ylabel('ln(asking price)')
        ax.text(.03,.03,f'n = {len(data[0]):,} / {len(data[1]):,}',transform=ax.transAxes,fontsize=8)
        for k in [0,1]:
            s=g.loc[g.accidents_or_damage==k,'price']
            accident.append(dict(model_line=line,accidents_or_damage=k,listings=len(s),median_price=s.median()))
    fig.suptitle('Recorded accident history within the same model family',fontsize=14,fontweight='bold',y=1.02)
    save(fig,'04_accident_boxplots.png','Accident versus no accident recorded','Compare accident histories within model families rather than across all brands. Age, mileage, trim and damage severity remain uncontrolled; unknown histories are excluded, and boxplot fliers are hidden only for readability.', '先在相同车型内比较事故记录。车龄、里程、配置和事故程度仍未控制；未知事故记录不纳入两组，箱线图仅为易读而隐藏散点。',pd.DataFrame(accident))
    brand=ev.groupby('manufacturer').agg(median_price=('price','median'),listing_count=('listing_id','size'),price_cut_share=('price_cut_flag','mean'),price_drop_missing_share=('price_drop_was_missing','mean')).reset_index().sort_values('median_price')
    fig,axes=plt.subplots(1,2,figsize=(12,7),sharey=True)
    pos=np.arange(len(brand)); axes[0].barh(pos,brand.median_price,color=COLORS[0]); axes[1].barh(pos,brand.listing_count,color=COLORS[1]); axes[0].set_yticks(pos,brand.manufacturer)
    axes[0].set(title='Median asking price',xlabel='USD');axes[0].xaxis.set_major_formatter(USD);axes[1].set(title='Listing count',xlabel='Listings')
    topbrand=brand.sort_values('listing_count',ascending=False).iloc[0]
    save(fig,'05_brand_price_and_count.png','Median price and listing count by brand',f'{topbrand.manufacturer} accounts for {int(topbrand.listing_count):,} listings ({100*topbrand.listing_count/len(ev):.1f}% of the cohort). Brand median prices also reflect differences in model mix, age and mileage; they are not rankings of resale-value retention.',f'{topbrand.manufacturer} 有 {int(topbrand.listing_count):,} 条，占样本 {100*topbrand.listing_count/len(ev):.1f}%。品牌中位价格也受车型、车龄和里程组成影响，不是保值率排名。',brand)
    cuts=brand.sort_values('price_cut_share')
    fig,ax=plt.subplots(figsize=(10.5,7))
    ax.barh(cuts.manufacturer,cuts.price_cut_share*100,color=COLORS[0]); ax.set(title='Share with a recorded price reduction, by brand',xlabel='Listings with price_drop > 0 (%)');ax.set_xlim(0,100)
    for y,(_,row) in enumerate(cuts.iterrows()): ax.text(row.price_cut_share*100+1,y,f'n={int(row.listing_count):,}',va='center',fontsize=8)
    total=ev.price_cut_flag.mean()*100
    save(fig,'06_price_cut_share_by_brand.png','Share of listings with a recorded price cut',f'{total:.1f}% of listings have a positive recorded price reduction. Missing price_drop values are treated as zero under the project assumption; this is neither a forecast of future cuts nor proof that a dealer originally overpriced a car.',f'{total:.1f}% 的记录有降价金额。按项目假设将缺失金额视为0；这既不是未来降价预测，也不能证明经销商一开始定价过高。',cuts)
    fields=['price','log_price','year','age','mileage','miles_per_year','seller_rating','driver_rating']
    stats=ev[fields].describe().T
    tab=stats[['count','mean','std','min','25%','50%','75%','max']]
    fig,ax=plt.subplots(figsize=(13,4.1));ax.axis('off')
    cell=[[f'{v:,.2f}' if c!='count' else f'{v:,.0f}' for c,v in row.items()] for _,row in tab.iterrows()]
    table=ax.table(cellText=cell,rowLabels=tab.index,colLabels=tab.columns,loc='center',cellLoc='right')
    table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,1.75)
    for (r,c),cl in table.get_celld().items():
        cl.set_edgecolor('#DCE4EC')
        if r==0:cl.set_facecolor('#173E60');cl.get_text().set_color('white')
        elif r%2==0:cl.set_facecolor('#F1F6FA')
    ax.set_title('Summary statistics | unfilled clean cohort',pad=15)
    save(fig,'07_summary_statistics.png','Summary statistics',f'The cohort contains {len(ev):,} listings. Median asking price is ${ev.price.median():,.0f}; median mileage is {ev.mileage.median():,.0f} miles. Missing values remain missing, and unusual observations remain in the handover.',f'样本共 {len(ev):,} 条，中位挂牌价 ${ev.price.median():,.0f}，中位里程 {ev.mileage.median():,.0f} 英里。交接数据保留缺失和异常记录。',stats.reset_index(names='field'))
    (root/'data/diagnostics/eda_takeaways.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
    print('Saved',len(results),'requested EDA figures',flush=True)
    return results
