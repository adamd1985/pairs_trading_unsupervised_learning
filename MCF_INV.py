import numpy as np
import numpy.linalg as LA
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as st
import scipy.optimize as sco
import bs4 as bs
import requests


def expRet(ret,n=252,com=False):
    """Calculates expected returns in one of two methods. Parameters are
    - ret: Pandas' DataFrame or Series of returns
    - n: optional, number of compounding periods in a year (252 for days, 52 for weeks, 12 for months, 4 for quarters...)
    - com: optional, deterimines whether expected returns are calculated as annualized sample mean or annualized compunded return"""
    if com:
        return (1+ret).prod()**(n/len(ret))-1
    else:
        return ret.mean()*n
    
def annualize_vol(ret, n=252):
    """Calculates volatility of sample returns. Parameters are:
    - ret: Pandas' DataFrame or Series of returns
    - n: optional, number of compounding periods in a year (252 for days, 52 for weeks, 12 for months, 4 for quarters...)"""
    return ret.std()*(n**0.5)

def sharpe(ret,n=252,rf=0,com=False):
    """Calculates Sharpe's ratio. Parameters are:
    - ret: Pandas' DataFrame or Series of returns
    - rf: optional, risk free rate (should be given as decimal, if it is omitted, we assume it to be zero)
    - n: optional, number of compounding periods in a year (252 for days, 52 for weeks, 12 for months, 4 for quarters...)
    - com: optional, deterimines whether expected returns are calculated as annualized sample mean or annualized compunded return"""
    return (expRet(ret,n,com)-rf)/annualize_vol(ret,n)

def hist(df,CDF=False):
    """For the given DataFrame or Series of returns function plot histogram along with appropriate normal curve for each colummn.
    Parameters are:
    - df: DataFrame or Series of returns.
    - CDF: If True is given returns comparison of empirical and theoretical CDF, while if False is given returns comparison of
    theoretical and empirical PDF"""
    if str(type(df))!="<class 'pandas.core.frame.DataFrame'>" and str(type(df))!="<class 'pandas.core.series.Series'>":
        return print("Wrong input! Please insert data as Pandas' Series or DataFrame object")
    elif str(type(df))=="<class 'pandas.core.series.Series'>":
        df=df.to_frame()
    else:
        pass
    n=df.shape[1]
    if n==1:
        Row=1
        fig = make_subplots(rows=1, cols=1) 
    elif n%2==0:
        Row=int(n/2)
        fig = make_subplots(rows=Row, cols=2,subplot_titles=df.columns,shared_yaxes=True,specs=[[{}, {}]]*Row) 
    else:
        Row=int((n-1)/2)+1
        fig = make_subplots(rows=Row, cols=2,subplot_titles=df.columns,
                            shared_yaxes=True,specs=[[{}, {}]]*(Row-1) +[[{"colspan": 2}, None]])
    l_Row=[item for sublist in [[i,i] for i in range(1,Row+1)] for item in sublist]
    
    for i in range(n):
        r=df.iloc[:,i]
        x_list=(np.linspace(min(r),max(r),100) if CDF else np.linspace(r.mean()-3*r.std(),r.mean()+3*r.std(),100))
        y_list=(st.norm.cdf(x_list,r.mean(),r.std()) if CDF else st.norm.pdf(x_list,r.mean(),r.std())/100)
        fig.add_trace(go.Histogram(x=r, marker=dict(color='Orange',line=dict(width=2,color='black')),
            histnorm='probability',cumulative=dict(enabled=CDF)),row=l_Row[i],col=(2 if (i+1)%2==0 else 1))
        fig.add_trace(go.Scatter(x=x_list,y=y_list,line_color='DarkGreen',fill='tozeroy'),
                      row=l_Row[i],col=(2 if (i+1)%2==0 else 1))
        fig.update_xaxes(zerolinecolor='black',row=l_Row[i],col=(2 if (i+1)%2==0 else 1))
        fig.update_yaxes(zerolinecolor='black',row=l_Row[i],col=(2 if (i+1)%2==0 else 1))
    fig.update_layout(title=dict(text='Histograms and normal curve',font=dict(size=30),x=0.5,y=0.95),showlegend=False)

    return fig.show()

def box(df):
    """For the given DataFrame of returns function plot box plot for each colummn.
    Function has only one required parameter - DataFrame or Series of returns."""
    if str(type(df))!="<class 'pandas.core.frame.DataFrame'>" and str(type(df))!="<class 'pandas.core.series.Series'>":
        return print("Wrong input! Please insert data as Pandas' Series or DataFrame object")
    elif str(type(df))=="<class 'pandas.core.series.Series'>":
        df=df.to_frame()
    else:
        pass
    fig=go.Figure()
    for i in range(df.shape[1]):
        r=df.iloc[:,i]
        IQR=r.quantile(q=0.75)-r.quantile(q=0.25)
        extremes=r[(r>=r.quantile(q=0.75)+3*IQR) | (r<=r.quantile(q=0.25)-3*IQR)].values
        fig.add_trace(go.Box(y=r,name=df.columns[i]))
        fig.add_trace(go.Scatter(x=[df.columns[i]]*len(extremes),y=extremes,
                                 mode='markers',marker=dict(color='black',symbol='star',size=15),showlegend=False))
    fig.update_layout(title=dict(text='Box plot of returns',font=dict(size=30),x=0.5,y=0.95),width=900,height=650)
    return fig.show()

def KS(df):
    """For the given DataFrame of returns function perform Kolmogorov-Smirnov test for each column.
    Function has only one required parameter - DataFrame or Series of returns."""
    if str(type(df))!="<class 'pandas.core.frame.DataFrame'>" and str(type(df))!="<class 'pandas.core.series.Series'>":
        return print("Wrong input! Please insert data as Pandas' Series or DataFrame object")
    elif str(type(df))=="<class 'pandas.core.series.Series'>":
        df=df.to_frame()
    else:
        pass
    return [st.kstest(df.iloc[:,i],st.norm(df.iloc[:,i].mean(),df.iloc[:,i].std()).cdf) for i in range(df.shape[1])]

def semidev(df,n=252,zeromean=False):
    """For the given DataFrame of returns function calculates annualized semideviation for each column. Paramters are:
    - df: DataFrame or Series of returns.
    - n: optional, annualization factor (252 for daily data, 12 for monthly...). If you don't font annualization, set it to 1.
    - zeromean: optional. If True: assumes that mean is zero. If False: function calculate mean from the data."""
    if str(type(df))!="<class 'pandas.core.frame.DataFrame'>" and str(type(df))!="<class 'pandas.core.series.Series'>":
        return print("Wrong input! Please insert data as Pandas' Series or DataFrame object")
    elif str(type(df))=="<class 'pandas.core.series.Series'>":
        df=df.to_frame()
    else:
        pass
    dfN=pd.DataFrame()
    for i in range(df.shape[1]):
        r=df.iloc[:,i]
        m=(0 if zeromean else r.mean())
        dfN[df.columns[i]]=[np.sqrt(np.mean((r[r<m]-m)**2))*(n**0.5)]
    dfN.index=['Semideviation']
    return dfN

def sortino(ret,n=252,rf=0,com=False):
    """Calculates Soritano's ratio. Parameters are:
    - ret: Pandas' DataFrame or Series of returns
    - rf: optional, risk free rate (should be given as decimal, if it is omitted, we assume it to be zero)
    - n: optional, number of compounding periods (annualization factor) in a year (252 for days, 52 for weeks, 12 for months...)
    - com: optional, deterimines whether expected returns are calculated as annualized sample mean or annualized compunded return"""
    return (expRet(ret,n,com)-rf)/semidev(ret,n).rename(index={'Semideviation':'Sortino'})

def drawdown(ret: pd.Series):
    """"Computes and returns a dataframe which contains: wealth index, previous peaks and relative drawdowns. Parameters are:
    - ret: time series of asset returns"""
    wealth_index = 1000*(1+ret).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({"Wealth":wealth_index,"Peaks": previous_peaks,"Drawdowns": drawdowns})

def VaR(df,a=0.01,hist_interpolation='linear'):
    """For the given DataFrame of returns function calculates VaR by three methods.
    Paramters are:
    -df: DataFrame or Series of returns.
    -a: optional. It is probability that loss can be higher than VaR.
    -hist_interpolation: sets estimator of quantile's position in historical VaR. For more info see help for Pandas quantile"""
    if any([isinstance(df,i) for i in [pd.Series, pd.DataFrame]])==False:
        return print("Wrong input! Please insert data as pd.Series or pd.DataFrame object")
    elif isinstance(df,pd.Series):
        df=df.to_frame()
    else:
        pass
    dfN=pd.DataFrame(index=['Historical','Gaussian','Cornish Fisher'],columns=df.columns)
    for i in range(df.shape[1]):
        r=df.iloc[:,i]
        z=st.norm().ppf(a)
        zk = (z +(z**2 - 1)*r.skew()/6 +(z**3 -3*z)*(r.kurtosis())/24 -(2*z**3 - 5*z)*(r.skew()**2)/36)
        dfN.iloc[0,i]=np.abs(r.quantile(a, interpolation=hist_interpolation))
        dfN.iloc[1,i]= -(r.mean() + z*r.std())
        dfN.iloc[2,i]= -(r.mean() + zk*r.std())
    return dfN

def cvar_historic(r, a=0.01,hist_interpolation='linear'):
    """Computes the Conditional VaR of Series or DataFrame"""
    if isinstance(r, pd.Series):
        is_beyond = r <= -VaR(r, a=a,hist_interpolation=hist_interpolation).iloc[0].values[0]
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, a=a)
    else:
        raise TypeError("Expected r to be a pd.Series or pd.DataFrame")

def per(W,ret=None,n=252,com=False,er_assumed=None):
    """Calculates expected returns of portfoli:
    - ret: data set of returns
    - W: iterable of weights
    - n: optional, number of compounding periods in a year (252 for days, 52 for weeks, 12 for months, 4 for quarters...)
    - com: optional, deterimines whether expected returns are calculated as annualized sample mean or annualized compunded retu
    - er_assumed: optional, assummed expected returns on each stock"""
    er=expRet(ret,n,com) if np.all(er_assumed==None) else er_assumed
    return er@W

def pV(W,ret=None,n=252,vol=False,cov_assumed=None):
    """Calculates variance returns of portfoli:
    - ret: data set of returns
    - W: iterable of weights
    - n: optional, number of compounding periods in a year (252 for days, 52 for weeks, 12 for months, 4 for quarters...)
    - vol: optional, determines whether you want to display portfolio's volatility (True) or variance (False)
    - cov_assumed: optional, assummed annualized covariance matrix of returns"""
    CovM=(ret.cov()*n if np.all(cov_assumed==None) else cov_assumed)
    return np.sqrt(W@CovM@W) if vol else W@CovM@W

def mvp(ret=None,f=252,com=False,cov_assumed=None):
    """Calculates MVP portfolio weights, volatility and expected returns for given data set of returns. Paramters are:
    - ret: optional, data set of returns. If it isn't provided, than it is required to provide er_assumed and cov_assumed.
    - f: optional, number of compounding periods in a year, i.e. frequency (252 for days, 52 for weeks, 12 for months, 4 for quarters...)
    - com: optional, deterimines whether expected returns are calculated as annualized sample mean or annualized compunded ret
    - cov_assumed: optional, assummed annualized covariance matrix of returns"""
    n=(len(ret.columns) if np.all(ret!=None) else len(cov_assumed))
    result=sco.minimize(lambda w: pV(w,ret,f,False,cov_assumed),[1/n]*n,constraints=[dict(type='eq',fun=lambda w:sum(w)-1)])
    return dict(w=result.x,er=per(result.x,ret,f,com),vol=np.sqrt(result.fun)) if np.all(ret!=None) else result.x

def targetP(ret=None,mi=None,Bounds=None,f=252,com=False,er_assumed=None, cov_assumed=None):
    """Calculates optimal portfolio weights, volatility and expected returns. Paramters are:
    - ret: optional, data set of returns. If it isn't provided, than it is required to provide er_assumed and cov_assumed.
    - mi: target level of expected returns. If mi is omitted, function will caulculate MVP weights!
    - f: optional, number of compounding periods in a year, i.e. frequency (252 for days, 52 for weeks, 12 for months, 4 for quarters...)
    - com: optional, deterimines whether expected returns are calculated as annualized sample mean or annualized compunded retu
    - er_assumed: optional, assummed expected returns on each stock
    - cov_assumed: optional, assummed annualized covariance matrix of returns"""
    n=(len(ret.columns) if np.all(ret!=None) else len(cov_assumed))
    if mi==None:
        result=sco.minimize(lambda w: pV(w,ret,f,False,cov_assumed),[1/n]*n,constraints=[dict(type='eq',fun=lambda w:sum(w)-1)])
        if np.all(ret!=None):
            return dict(w=result.x, er=per(result.x,ret,f,com,er_assumed),vol=np.sqrt(result.fun))
        else:
            return dict(w=result.x,er=(None if np.all(er_assumed==None) else (result.x)@er_assumed),vol=np.sqrt(result.fun))
    elif type(mi)==float or type(mi)==np.float64:
        result=sco.minimize(lambda w: pV(w,ret,f,False,cov_assumed),[1/n]*n,bounds=(None if Bounds==None else [Bounds]*n),
                        constraints=[dict(type='eq',fun=lambda w:sum(w)-1),
                                     dict(type='eq',fun=lambda w:per(w,ret,f,com,er_assumed)-mi)])
        return dict(w=result.x, er=mi,vol=np.sqrt(result.fun))
    else:
        return print('\33[91mWrong input ginven for parameter mi!\33[0m')
    
def EF(ret=None,Range=[0.01,0.4],plot=None,f=252,com=False,er_assumed=None,cov_assumed=None):
    """Prepares data for plotting efficient frontier or plot this curve (depending on what user chooses). Paramters are:
    - ret: optional, data set of returns. If it isn't provided, than it is required to provide er_assumed and cov_assumed.
    - Range: optional, range in which you want to see efficient frontier.
    - f: optional, number of compounding periods in a year, i.e. frequency (252 for days, 52 for weeks, 12 for months, 4 for quarters...)
    - com: optional, deterimines whether expected returns are calculated as annualized sample mean or annualized compunded retu
    - plot: optional, determines whether you want to plot efficient frontier or not. Possible values are:
        a) None - on values will be displayed
        b) 'curve' - only curve will be plotted
        c) 'mvp' - curve with MVP will be plotted
        d) 'full' - curve with MVP will be plotted and efficient part will be emphasised
    - er_assumed: optional, assummed expected returns on each stock
    - cov_assumed: optional, assummed annualized covariance matrix of returns"""
    mi=np.arange(Range[0],Range[1]+0.01,0.01)
    sigma=np.array([targetP(ret,m,f=f,com=com,er_assumed=er_assumed,cov_assumed=cov_assumed)['vol'] for m in mi])
    if plot==None:
        return dict(sigma=sigma,er=mi)
    elif plot=='curve':
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=sigma,y=mi,line=dict(color='Blue',width=3)))
        fig.update_layout(xaxis=dict(title_text='\$\sigma$',range=[-0.01,max(0.25,np.max(sigma)+0.01)],zerolinecolor='Black'),
                  yaxis=dict(title_text='$\mu$',zerolinecolor='Black'),
                  title=dict(text="Efficient frontier",x=0.5,y=0.87,font=dict(size=25,color='Navy')))
        return fig
    elif plot=='mvp':
        M=targetP(ret,m,f=f,com=com,er_assumed=er_assumed,cov_assumed=cov_assumed)
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=sigma,y=mi,line=dict(color='Blue',width=3)))
        fig.add_trace(go.Scatter(x=[M['vol']],y=[M['er']],marker=dict(color='Red',line=dict(color='Black',width=2),size=10)))
        fig.update_layout(xaxis=dict(title_text='\$\sigma$',range=[-0.01,max(0.25,np.max(sigma)+0.01)],zerolinecolor='Black'),
                  yaxis=dict(title_text='$\mu$',zerolinecolor='Black'),showlegend=False,
                  title=dict(text="Efficient frontier",x=0.5,y=0.87,font=dict(size=25,color='Navy')))
        return fig
    elif plot=='full':
        M=targetP(ret,f=f,com=com,er_assumed=er_assumed,cov_assumed=cov_assumed)
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=sigma,y=mi,line=dict(color='Blue',dash='dash'),fill="toself",fillcolor='Pink'))
        fig.add_trace(go.Scatter(x=sigma[mi>M['er']],y=mi[mi>M['er']],line=dict(color='Blue',width=3)))
        fig.add_trace(go.Scatter(x=[M['vol']],y=[M['er']],marker=dict(color='Red',line=dict(color='Black',width=2),size=10)))
        fig.update_layout(xaxis=dict(title_text='\$\sigma$',range=[-0.01,max(0.25,np.max(sigma)+0.01)],zerolinecolor='Black'),
                  yaxis=dict(title_text='$\mu$',zerolinecolor='Black'),showlegend=False,
                  title=dict(text="Efficient frontier",x=0.5,y=0.87,font=dict(size=25,color='Navy')))

        return fig
    else:
        return print("\33[91mWrong input ginven for parameter plot. Expected values are: None,'curve','mvp' or 'full'\33[0m") 
    
def maxSharpe(ret=None,rf=0.0,Bounds=[-1,1],f=252,com=False,er_assumed=None,cov_assumed=None):
    """Calculates weights for portfolio with maximal Sharpe's ratio, its volatility and expected returns. Paramters are:
    - ret: optional, data set of returns on risky assets. If it isn't provided, than it is required to provide er_assumed and cov_assumed.
    - f: optional, number of compounding periods in a year, i.e. frequency (252 for days, 52 for weeks, 12 for months, 4 for quarters...)
    - com: optional, deterimines whether expected returns are calculated as annualized sample mean or annualized compunded retu
    - rf: data set of returns on risk-free assets or assumed constant 
    - er_assumed: optional, assummed expected returns on each stock
    - cov_assumed: optional, assummed annualized covariance matrix of returns"""
    n=(len(ret.columns) if np.all(ret!=None) else len(cov_assumed))
    r_f=(rf if type(rf)==float or type(rf)==np.float64 else expRet(rf))
    result=sco.minimize(lambda w: -(per(w,ret,f,com,er_assumed)-r_f)/pV(w,ret,f,True,cov_assumed),[1/n]*n,bounds=[Bounds]*n,
                        constraints=[dict(type='eq',fun=lambda w:sum(w)-1)]) 
    return dict(w=result.x, er=per(result.x,ret,f,com,er_assumed),vol=pV(result.x,ret,f,True,cov_assumed),sharpe=-result.fun)

def portfolio_tracking_error(weights, ref_r, bb_r):
    """Returns the tracking error between the reference returns and a portfolio of building block returns held with given weights.
    Parameters are:
    1. weights - data set of portfolio weights.
    2. ret_r - data set of target returns
    3. bb_r - data set of explanatory returns
    All parameters should be given as np.array, pd.Series or pd.DataFrame."""
    return np.sqrt(((ref_r - (weights*bb_r).sum(axis=1))**2).sum())

def style_analysis(dependent_variable, explanatory_variables):
    """Returns the optimal weights that minimizes the Tracking error between a portfolio of the explanatory variables and the
    dependent variable. Parameters are:
    1. dependent_variable - data set of target returns
    2. explanatory_variables - data set of explanatory returns
    All parameters should be given as np.array, pd.Series or pd.DataFrame."""
    n=len(explanatory_variables.columns)
    result=sco.minimize(lambda w: portfolio_tracking_error(w,dependent_variable, explanatory_variables),
                        [1/n]*n,bounds=[(0,1)]*n,constraints=[dict(type='eq',fun=lambda w:sum(w)-1)])
    return pd.Series(result.x, index=explanatory_variables.columns)

def compound(r):
    """returns the result of compounding the set of returns in r"""
    return np.expm1(np.log1p(r).sum())

def weight_ew(r, cap_weights=None, max_cw_mult=None, microcap_threshold=None, **kwargs):
    """Returns the weights of the EW portfolio based on the asset returns "r" as a DataFrame
    If supplied a set of capweights and a capweight tether, it is applied and reweighted """
    n = len(r.columns)
    ew = pd.Series(1/n, index=r.columns)
    if cap_weights is not None:
        cw = cap_weights.loc[r.index[0]] # starting cap weight
        ## exclude microcaps
        if microcap_threshold is not None and microcap_threshold > 0:
            microcap = cw < microcap_threshold
            ew[microcap] = 0
            ew = ew/ew.sum()
        #limit weight to a multiple of capweight
        if max_cw_mult is not None and max_cw_mult > 0:
            ew = np.minimum(ew, cw*max_cw_mult)
            ew = ew/ew.sum() #reweight
    return ew

def weight_cw(r, cap_weights, **kwargs):
    """Returns the weights of the CW portfolio based on the time series of capweights"""
    return cap_weights.loc[r.index[0]]

def weight_mvp(ret,cov,Bounds=None,**kwargs):
    """Calculates MVP weights for given data set of returns and covariance estimator function. Paramters are:
    - ret: data set of returns
    - cov: function for covariance estimation
    - Bounds: optional, determines upper and lower bound between which weights can take values"""
    n=len(ret.columns)
    CovM=cov(ret,**kwargs)
    result=sco.minimize(lambda w: w@CovM@w,[1/n]*n,bounds=(None if Bounds==None else [Bounds]*n),
                        constraints=[dict(type='eq',fun=lambda w:sum(w)-1)])
    return result.x

def sample_cov(r, **kwargs):
    """Estimates the sample covariance of the supplied returns. Parameters are:
    - r: data set of returns"""
    return r.cov()

def cc_cov(r, **kwargs):
    """Estimates a covariance matrix by using the Elton/Gruber Constant Correlation model. Parameters are:
    - r: data set of returns"""
    rhos = r.corr()
    n = rhos.shape[0]
    rho_bar = (rhos.values.sum()-n)/(n*(n-1))
    ccor = np.full_like(rhos, rho_bar) # This creates a matrix with all elements equal to rho_bar
    np.fill_diagonal(ccor, 1.) # Now we fill the diagonal with 1
    sd = r.std()
    return pd.DataFrame(np.diag(sd)@ccor@np.diag(sd), index=r.columns, columns=r.columns)

def shrinkage_cov(r, prior=cc_cov, delta=0.5, **kwargs):
    """Covariance estimator that shrinks between the Sample Covariance and the Constant Correlation Estimators. Parameters are:
    - r: data set of returns
    - delta: shrinkage parameter
    - prior: function which estimates covariance matrix according to some model. By default we use constant correlation model"""
    return delta*prior(r, **kwargs) + (1-delta)*sample_cov(r, **kwargs)

def backtest_ws(r, window=60, weighting=weight_ew,**kwargs):
    """Backtests a given weighting scheme, given some parameters:
    - r : asset returns to use to build the portfolio
    - estimation_window: the window to use to estimate parameters
    - weighting: the weighting scheme to use, must be a function that takes "r", and a variable number of keyword arguments """
    weights =pd.DataFrame([weighting(r.iloc[i:i+window],**kwargs) for i in range(len(r)-window)],
                       index=r.iloc[window:].index,columns=r.columns)
    return (weights * r).sum(axis="columns",  min_count=1) #mincount is to generate NAs if all inputs are NAs

def summary_stats(r, n=252,com=False, rf=0.03):
    """Return a DataFrame that contains aggregated summary stats for the returns in the columns of r. Parameters are:
    - r: data set of returns
    - n: optional, number of periods within a year
    - com: optional, deterimines whether expected returns are calculated as annualized sample mean or annualized compunded return
    - rf: optional, determines risk-free rate"""
    return pd.DataFrame({"Annualized Return": expRet(r,n,com),
                        "Annualized Vol": annualize_vol(r,n),
                        "Skewness": r.skew(),
                        "Kurtosis": r.kurt(),
                        "Cornish-Fisher VaR (5%)": VaR(r,a=0.05).loc['Cornish Fisher'],
                        "Historic CVaR (5%)": cvar_historic(r),
                        "Sharpe Ratio": sharpe(r,n,rf,com),
                        "Max Drawdown": r.aggregate(lambda r: drawdown(r).Drawdowns.min())})

def implied_returns(delta, sigma, w):
    """Obtain the implied expected returns by the market - i.e. construct vector Pi. Parameters are:
    - delta: Risk Aversion Coefficient (scalar)
    - sigma: Variance-Covariance Matrix (N x N) as pd.DataFrame
    - w: Portfolio weights (N x 1) as pd.Series
    Returns an N x 1 vector of Returns as Series """
    ir = delta * sigma.dot(w).squeeze() # to get a series from a 1-column dataframe
    ir.name = 'Implied Returns'
    return ir

def proportional_prior(sigma, tau, p):
    """Returns the He-Litterman simplified Omega. Parameters are:
    - sigma: N x N Covariance Matrix as pd.DataFrame
    - tau: a scalar
    - p: a K x N pd.DataFrame linking Q and Assets
    returns a P x P pd.DataFrame, a Matrix representing Prior Uncertainties"""
    helit_omega = p.dot(tau * sigma).dot(p.T)
    return pd.DataFrame(np.diag(np.diag(helit_omega.values)),index=p.index,columns=p.index) 

def bl(w_prior, sigma_prior, p, q,omega=None,delta=2.5, tau=.02):
    """Computes the posterior expected returns based on the original black litterman reference model. Parameters are:
    - W_prior: must be an N x 1 vector of weights, a pd.Series
    - Sigma_prior: is an N x N covariance matrix, a pd.DataFrame
    - P: must be a K x N matrix linking Q and the Assets, a pd.DataFrame
    - Q: must be an K x 1 vector of views, a pd.Series
    - Omega: must be a K x K matrix a pd.DataFrame, or None (in that case He-Litterman simplified Omega is used)
    - delta and tau: are scalars""" 
    if omega is None: omega = proportional_prior(sigma_prior, tau, p) #if omega isn't specified use He-Litterman siplification
    N = w_prior.shape[0] # How many assets do we have?
    K = q.shape[0] # And how many views?
    pi = implied_returns(delta, sigma_prior,  w_prior) # Determine market implied expected returns - Pi
    sigma_prior_scaled = tau * sigma_prior  # Adjust (scale) Sigma by the uncertainty scaling factor
    # estimating expected returns and covariance matrix by B-L model
    mu_bl = pi + sigma_prior_scaled.dot(p.T).dot(LA.inv(p.dot(sigma_prior_scaled).dot(p.T) + omega).dot(q - p.dot(pi).values))
    sigma_bl=sigma_prior+sigma_prior_scaled-sigma_prior_scaled.dot(p.T).dot(LA.inv(p.dot(sigma_prior_scaled).dot(p.T)+omega)).dot(p).dot(sigma_prior_scaled)
    return (mu_bl, sigma_bl)

def w_msr(sigma, mu,rf=0):
    """Optimal (Tangent/Max Sharpe Ratio) Portfolio weights by using the Markowitz Optimization Procedure. Parameters are:
    - Mu: the vector of Excess expected returns given as pd.Series
    - Sigma: covariance matrix. Must be given as N x N matrix in pd.DataFrame"""
    invS=pd.DataFrame(LA.inv(sigma.values), index=sigma.columns, columns=sigma.index)
    w = invS.dot(mu-rf)
    return w/sum(w)

def ewma(x,alpha):
    """This function implements EMWA using a recursive algorithm. Parameters are:
    - x: is data frame of values
    - alpha: is EWMA parameter alpha, which is equal to 1-lambda"""
    y = np.zeros_like(x)
    y[0]=x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1-alpha)* y[i-1]
    return y

def weight_mvp_ENC(ret,ENC,cov=sample_cov,Bounds=None,**kwargs):
    """Calculates MVP weights with ENC constraint for given data set of returns and covariance estimator function. Paramters are:
    - ret: data set of returns
    - ENC: lower bound for ENC
    - cov: function for covariance estimation
    - Bounds: optional, determines upper and lower bound between which weights can take values"""
    n=len(ret.columns)
    if ENC<1 or ENC>n:
        return print("'\33[91mENC has to be in interval from 1 to N!\33[0m")
    CovM=cov(ret,**kwargs)
    result=sco.minimize(lambda w: w@CovM@w,[1/n]*n,bounds=(None if Bounds==None else [Bounds]*n),
                        constraints=[dict(type='eq',fun=lambda w:sum(w)-1),dict(type='ineq',fun=lambda w:1/sum(w**2)-ENC)])
    return result.x

def weight_mdp(ret,cov=sample_cov,Bounds=None,**kwargs):
    """Calculates MDP weights for given data set of returns and covariance estimator function. Paramters are:
    - ret: data set of returns
    - cov: function for covariance estimation
    - Bounds: optional, determines upper and lower bound between which weights can take values"""
    n=len(ret.columns)
    CovM=cov(ret,**kwargs)
    v=np.mean(np.sqrt(np.diag(CovM)))**2
    rho=np.corrcoef(ret,rowvar=False)
    result=sco.minimize(lambda w: v*w@rho@w,[1/n]*n,bounds=(None if Bounds==None else [Bounds]*n),
                        constraints=[dict(type='eq',fun=lambda w:sum(w)-1)])
    return result.x

def risk_contribution(ret,w):
    """Compute the contributions to risk of portfolio constituents, given a data set of returns and portfolio weights
    - ret: data set of returns
    - w: list or np.array of portfolio weights"""
    covM=ret.cov()
    return (covM@w)*np.array(w).T/(w@covM@w)

def target_risk_contributions(ret,target_risk):
    """Returns the weights of the portfolio with the contributions to portfolio risk are as close as possible to the 
    target_risk. Parameters are:
    - ret: data set of returns
    - target_risk: list or np.array with target level of risk contribution
    - Bounds: bounds for weights in which you want to perform optimizaiton"""
    n = ret.shape[1] 
    return sco.minimize(lambda w: ((risk_contribution(ret, w)-target_risk)**2).sum(), [1/n]*n,
                           bounds=[(0,1)]*n,method='SLSQP',
                           constraints=({'type': 'eq','fun': lambda w: np.sum(w) - 1})).x

def equal_risk_contributions(ret):
    """Returns the weights of the portfolio that equalizes the risk contributions of the constituents. Parameters are:
    - ret: data set of returns
    - Bounds: bounds for weights in which you want to perform optimizaiton"""
    n = ret.shape[1]
    return target_risk_contributions(ret,np.array([1/n]*n))

def run_cppi(risky_r, safe_r=None, m=3, start=1000, floor=0.8, riskfree_rate=0.03,f=12, drawdown=None):
    """Run a backtest of the CPPI strategy, given a set of returns for the risky asset. Returns a dictionary containing: Asset
    Value History, Risk Budget History, Risky Weight History... Parameters are:
    - riksy_r: data set of returns on risky asset(s)
    - safe_r: optional, data set of returns on safe assets, if None is given, risk-free rate is assumed as safe return
    - m: optional, multiplier in your strategy
    - start: optional, your initial investment
    - floor: optional, floor rate (fraction of your initial investment which you want to preserve at all costs)
    - riskfree_rate: optinal, constant risky-free rate in your economy
    - f: optional, frequency of your data (252 for daily, 12 for monthly, 1 for annually...)
    - drawdown: optional, maximal acceptable drawdown, if None drawdown is not used in strategy implementation"""
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start*floor
    peak = account_value
    if isinstance(risky_r, pd.Series): risky_r=risky_r.to_frame('R')
    #treatment of safe returns
    safe_ret = pd.DataFrame().reindex_like(risky_r)
    if isinstance(safe_r,pd.DataFrame):
        for i in range(risky_r.shape[1]):
            safe_ret.iloc[:,i]=safe_r
        safe_ret.fillna(method='bfill',inplace=True)
    else:
        safe_ret.values[:] = riskfree_rate/f # fast way to set all values to a number

    account_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    floorval_history = pd.DataFrame().reindex_like(risky_r)
    peak_history = pd.DataFrame().reindex_like(risky_r)

    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak*(1-drawdown)
        cushion = (account_value - floor_value)/account_value
        risky_w = np.maximum( np.minimum(m*cushion, 1), 0)
        safe_w = 1-risky_w
        risky_alloc = account_value*risky_w
        safe_alloc = account_value*safe_w
        account_value = risky_alloc*(1+risky_r.iloc[step]) + safe_alloc*(1+safe_ret.iloc[step])
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value
        floorval_history.iloc[step] = floor_value
        peak_history.iloc[step] = peak
    risky_wealth = start*(1+risky_r).cumprod()
    backtest_result = {"Wealth": account_history,"Risky Wealth": risky_wealth, "Risk Budget": cushion_history,
        "Risky Allocation": risky_w_history,"m": m,"start": start,"floor rate": floor,"risky_r":risky_r,
        "safe_r": safe_r,"drawdown": drawdown,"peak": peak_history,"floor": floorval_history}
    return backtest_result

def gbm(n_years = 10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, s_0=100.0, prices=True):
    """Function returns evolution of Geometric Brownian Motion trajectories through Monte Carlo simulations. Parmeters are:
    - n_years:  The number of years to generate data for
    - n_paths: The number of scenarios/trajectories
    - mu: Annualized Drift, e.g. expected return on risky asset
    - sigma: Annualized Volatility
    - steps_per_year: Granularity of the simulation
    - s_0: Initial value
    - prices: Boolean, determines wheter you want to obtain prices (GBM) or returns (ABM)"""
    dt = 1/steps_per_year
    n_steps = int(n_years*steps_per_year) + 1
    rets_plus_1 = np.random.normal(loc=(1+mu*dt), scale=(sigma*np.sqrt(dt)), size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1 
    return s_0*pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1-1

def pv(l, r):
    """Compute the present value of a list of liabilities given by the time (as an index). Parameters are:
    -l: pd.Series or pd.DataFrame of liabilities
    -r: constant discount rate"""
    if type(l)==pd.Series or type(l)==pd.DataFrame:
        return ((1+r)**(-l.index))@l if type(r)!=pd.Series and type(r)!=pd.DataFrame else pd.Series([((1+i)**(-l.index))@l for i in r],index=r.index)
    else:
        print('l has to be given with Pandas')

def funding_ratio(assets, liabilities, r):
    """Computes the funding ratio of a series of liabilities, based on an interest rate and current value of assets. Parameters are:
    -assets: current value of assets, or series of their future cash flows given in pd.Series or pd.DataFrame
    -liabilities: pd.Series or pd.DataFrame of liabilities
    -r: constant discount rate"""
    return assets/pv(liabilities,r)if type(assets)!=pd.Series and type(assets)!=pd.DataFrame else pv(assets,r)/pv(liabilities,r)

def cir(n_years = 10, n_scenarios=1, a=0.05, b=0.03, sigma=0.05, steps_per_year=12, r_0=None, Return='r'):
    """Generate random interest rate evolution over time using the CIR model. Parameters are:
    -n_years: optional, number of years in your simulation
    -n_scenarios: optional, number of simulations to be created
    -a: optional, speed of mean reversion
    -b: optional, is assumed long run average annualized rate, not the short rate
    -r_0: optional, is assumed initial annualized rate, not the short rate
    -sigma: optional, volatility of interest rates
    -steps_per_year: optional, number of compoundings withing one year
    -Return: optional, determine what do you want to see as output: p- prices, r-rates or b-both"""
    if Return not in ['b','p','r']: return print('Wrong input given as parameter Returns')
    if r_0 is None: r_0 = b 
    r_0,dt,num_steps = np.log1p(r_0),1/steps_per_year,int(n_years*steps_per_year) + 1 
    shock = np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
    rates = np.empty_like(shock)
    rates[0] = r_0

    if Return=='r':
        for step in range(1, num_steps):
            r_t = rates[step-1]
            rates[step] = abs(r_t + a*(b-r_t)*dt + sigma*np.sqrt(r_t)*shock[step])
    else:
        h = np.sqrt(a**2 + 2*sigma**2)
        prices = np.empty_like(shock)
        def price(ttm, r):
            _A = ((2*h*np.exp((h+a)*ttm/2))/(2*h+(h+a)*(np.exp(h*ttm)-1)))**(2*a*b/sigma**2)
            _B = (2*(np.exp(h*ttm)-1))/(2*h+(h+a)*(np.exp(h*ttm)-1))
            return _A*np.exp(-_B*r)
        prices[0] = price(n_years, r_0)
        for step in range(1, num_steps):
            r_t = rates[step-1]
            rates[step] = abs(r_t + a*(b-r_t)*dt + sigma*np.sqrt(r_t)*shock[step])
            prices[step] = price(n_years-step*dt, rates[step])
        prices = pd.DataFrame(data=prices, index=range(num_steps))
        
    rates = pd.DataFrame(data=np.expm1(rates), index=range(num_steps))
    return rates if Return=='r' else (prices if Return=='p' else (rates,prices))

def bond_cash_flows(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12):
    """Returns the series of cash flows generated by a bond, indexed by the payment/coupon number. Parameters are:
    -maturity: time at which bond expire (in years)
    -principal: optional, amount that has to be returned at the end of trading period
    - coupon_rate: optional, interest rate which has to be paid to bond owner (decimal)
    - coupons_per_year: number of coupons paid per year (1 for yearly, 2 for semiannually, 12 for monthly...)"""
    cash_flows = pd.Series((principal*coupon_rate/coupons_per_year), index=np.arange(1, round(maturity*coupons_per_year)+1)) 
    cash_flows.iloc[-1] += principal 
    return cash_flows

def bond_price(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12, discount_rate=0.03):
    """Computes the price of a coupon bond. Parameters are:
    -maturity: time at which bond expire (in years)
    -principal: optional, amount that has to be returned at the end of trading period
    - coupon_rate: optional, interest rate which has to be paid to bond owner (decimal)
    - coupons_per_year: number of coupons paid per year (1 for yearly, 2 for semiannually, 12 for monthly...)
    -discount_rate: optional, rate which is used to discount cash flows"""
    if isinstance(discount_rate, pd.DataFrame): 
        prices = pd.DataFrame(index=discount_rate.index, columns=discount_rate.columns)
        for t in discount_rate.index:
            prices.loc[t]=bond_price(maturity-t/coupons_per_year,principal,coupon_rate,coupons_per_year,discount_rate.loc[t])
        return prices
    else: # base case ... single time period
        if maturity <= 0: return principal*(1+coupon_rate/coupons_per_year)
        return pv(bond_cash_flows(maturity, principal, coupon_rate, coupons_per_year), discount_rate/coupons_per_year)

def macaulay_duration(flows, discount_rate):
    """Computes the MacCaulay Duration of a sequence of cash flows, given a per-period discount rate. Parameters are:
    - flows: pd.Series or pd.DataFrame of bonds cashflows
    -discount_rate: yield to maturity or per-period discount rate"""
    if type(flows)!=pd.DataFrame and type(flows)!=pd.Series: return print('Argument flows has to be defined with Pandas')
    dcf = ((1+discount_rate)**(-flows.index))*flows
    return (flows.index*dcf).sum()/dcf.sum()

def convexity(flows, discount_rate):
    """Computes the Convexity of a sequence of cash flows, given a per-period discount rate. Parameters are:
    -flows: pd.Series or pd.DataFrame of bonds cashflows
    -discount_rate: yield to maturity or per-period discount rate"""
    if type(flows)!=pd.DataFrame and type(flows)!=pd.Series: return print('Argument flows has to be defined with Pandas')
    t=flows.index
    dcf = ((1+discount_rate)**(-t))*flows
    return ((t+1)*t*dcf).sum()/dcf.sum()/((1+discount_rate)**2)

def match_durations(cf_t, cf_s, cf_l, discount_rate):
    """Returns the weight W in cf_s that, along with (1-W) in cf_l will have an effective duration that matches cf_t.Parameters:
    -cf_t: target cash flows
    -cf_s: cash flows of bond with shorter maturity
    -cf_l: cash flows of bond with longer maturity"""
    d_t = macaulay_duration(cf_t, discount_rate)
    d_s = macaulay_duration(cf_s, discount_rate)
    d_l = macaulay_duration(cf_l, discount_rate)
    return (d_l - d_t)/(d_l - d_s)

def bond_total_return(prices, principal, coupon_rate, cpy=12,pf=12):
    """Computes the total return of a Bond based on bond prices and coupon payments. Calculation assumes that coupons are paid
    out at the end of the period and that they are reinvested in the bond. Parameters are:
    -prices:pd.Series of pd.DataFrame of bond prices
    -principal: face value of the bond
    -coupon_rate: interest rate paid to bond owner (in decimal notation)
    -cpy: optional, number of coupons per year
    -pf: optional, price frequency (12 if series contain monthly prices, 1 for annual prices and so on)"""
    if type(prices)!=pd.DataFrame and type(prices)!=pd.Series: return print('prices has to be given as DataFrame or Series')
    if type(prices)==pd.Series: prices=prices.to_frame('TR')
    coupons = pd.DataFrame(data = 0, index=prices.index, columns=prices.columns)
    t_max = prices.index.max()
    pay_date = np.linspace(pf/cpy, t_max, int(cpy*t_max/pf), dtype=int)
    coupons.iloc[pay_date] = principal*coupon_rate/cpy
    return ((prices + coupons)/prices.shift()-1).dropna()

def bt_mix(r1, r2, allocator, **kwargs):
    """Runs a back test (simulation) of allocating between a two sets of returns. Parameters are:
    - r1 and r2: are T x N DataFrames or returns where T is the time step index and N is the number of scenarios.
    - allocator: is a function that takes two sets of returns and allocator specific parameters, and produces
    an allocation to the first portfolio (the rest of the money is invested in the LHP) as a T x 1 DataFrame"""
    if not r1.shape == r2.shape: raise ValueError("r1 and r2 should have the same shape")
    weights = allocator(r1, r2, **kwargs)
    if not weights.shape == r1.shape: raise ValueError("Allocator returned weights with a different shape than the returns")
    return weights*r1 + (1-weights)*r2

def fixedmix_allocator(r1, r2, w1, **kwargs):
    """Produces a time series over T steps of allocations between the PSP and GHP across N scenarios
    -r1 and r2 are returns on PSP and LHP given as T x N DataFrames 
    -w1 is fixed weight put in in portfolio 1"""
    return pd.DataFrame(data = w1, index=r1.index, columns=r1.columns)

def glidepath_allocator(r1, r2, start_glide=1, end_glide=0.0):
    """Allocates weights to r1 starting at start_glide and gradually moving to the end weight at end_glide. Parameters are:
    -r1 and r2 are returns on PSP and LHP given as T x N DataFrames 
    -start_glide: weight from which gliding starts
    -end_glide: weight with whih gliding ends"""
    n_points,n_col = r1.shape # This are the number of points in time and the number of paths
    path = pd.Series(np.linspace(start_glide, end_glide, n_points))
    return pd.DataFrame([]*n_col, axis=1,index=r1.index,columns=r1.columns)

def floor_allocator(psp_r, ghp_r, floor, zc_prices, m=3):
    """Allocate between PSP and LHP with the goal to provide exposure to the upside of the PSP without violating the floor.
    It uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple of the cushion in the PSP. Parameters are:
    -psp_r: pd.Series or pd.DataFrame of returns on PSP 
    -ghp_r: pd.Series or pd.DataFrame of returns on GHP
    -floor: floor which you don't want to breach
    -zc_prices: pd.Series or pd.DataFrame of zero coupon bonds prices
    -m: optional, multiplier in CPPI strategy"""
    if all([isinstance(i,pd.Series) or isinstance(i,pd.DataFrame) for i in [psp_r,ghp_r,zc_prices]])==False:
        return print('Wrong input! Returns and prices have to be givene as pd.Series or pd.DataFrame')
    if zc_prices.shape != psp_r.shape: raise ValueError("PSP and ZC Prices must have the same shape")
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1, n_scenarios) # I initialize each scenario with 1
    floor_value = np.repeat(1, n_scenarios) # I set my floor value to one. But then update
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    for step in range(n_steps):
        floor_value = floor*zc_prices.iloc[step] # PV of Floor assuming today's rates and flat YC
        cushion = (account_value - floor_value)/account_value
        psp_w = (m*cushion).clip(0, 1) # same as applying min and max
        # This tells me how much to invest in LHP (sometimes called goal hedging portfolio)
        ghp_w = 1-psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        account_value=psp_alloc*(1+psp_r.iloc[step])+ghp_alloc*(1+ghp_r.iloc[step])# recompute the new account value 
        w_history.iloc[step] = psp_w
    return w_history

def drawdown_allocator(psp_r, ghp_r, maxdd, m=3):
    """Allocate between PSP and GHP with the goal to provide exposure to the upside of the PSP without going violating the
    floor. Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple of the cushion in the PSP. Parameters are:
    -psp_r: pd.Series or pd.DataFrame of returns on PSP 
    -ghp_r: pd.Series or pd.DataFrame of returns on GHP
    -maxdd: pd.Series or pd.DataFrame of zero coupon bonds prices
    -m: optional, multiplier in CPPI strategy"""
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1, n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    peak_value = np.repeat(1, n_scenarios)
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    for step in range(n_steps):
        floor_value = (1-maxdd)*peak_value ### Floor is based on Prev Peak
        cushion = (account_value - floor_value)/account_value
        psp_w = (m*cushion).clip(0, 1) # same as applying min and max
        ghp_w = 1-psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        peak_value = np.maximum(peak_value, account_value)
        w_history.iloc[step] = psp_w
    return w_history

def terminal_stats(rets, floor = 0.8, cap=np.inf, name="Stats"):
    """Produce Summary Statistics on the terminal values per invested dollar across a range of N scenarios. Parameters are:
    -rets: a T x N DataFrame of returns, where T is the time-step (we assume rets is sorted by time)
    -floor: minimum acceptable asset to cover liability
    -cap: maximal acceptable asset to cover liability
    -name: name which you want to assign to column with statistics"""
    terminal_wealth = (rets+1).prod()
    breach = terminal_wealth < floor
    reach = terminal_wealth >= cap
    p_breach = breach.mean() if breach.sum() > 0 else np.nan
    p_reach = breach.mean() if reach.sum() > 0 else np.nan
    e_short = (floor-terminal_wealth[breach]).mean() if breach.sum() > 0 else np.nan
    e_surplus = (cap-terminal_wealth[reach]).mean() if reach.sum() > 0 else np.nan
    return pd.DataFrame.from_dict({"mean": terminal_wealth.mean(),"std" : terminal_wealth.std(),"p_breach": p_breach,
        "e_short":e_short,"p_reach": p_reach,"e_surplus": e_surplus}, orient="index", columns=[name])

def save_sp500_tickers():
    """Download ticker symbols of current S&P500 companies as list of strings. No arguments are required."""
    soup = bs.BeautifulSoup(requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies').text, 'html')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        if not '.' in ticker:
            tickers.append(ticker.replace('\n',''))
    return tickers

