



# 使用交叉验证衡量不同模型的准确度及稳定性，从而为选择模型提供依据
def cv_model_selection(models,names,x,y,cv=10):
    means,stds = [],[]
    for model in models:
        model.fit(x,y)
        res_array = cross_val_score(model,x,y,cv=cv)
        means.append(res_array.mean())
        stds.append(res_array.std())
    res_df = pd.DataFrame({'model_name':names,'mean':means,'std':stds})
    res_df['mean/std'] = res_df['mean']/res_df['std']
    res_df = res_df.sort_values('mean/std',ascending=False)
    return res_df


# 使用随机森林算法判断特征的相对重要性
def feature_importance(df,target,show_plot=False):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    y = df[target]
    x = df.drop(target,axis=1)
    model.fit(x,y)
    res_df = pd.DataFrame({'feature':x.columns,'importance':model.feature_importances_})
    res = res_df.sort_values('importance',ascending=False)
    res['cum_importance'] = res.importance.cumsum()
    if show_plot == True:
        plt.subplot(121)
        sns.barplot(res.feature,res.importance)
        plt.subplot(122)
        plt.plot(np.arange(1,res.shape[0]+1),res.cum_importance,linewidth=2)
    return(res)