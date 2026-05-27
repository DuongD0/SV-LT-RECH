# Deep learning enhanced volatility modeling with covariates

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//c41241c9-a2d3-4f21-886c-204b1cff21fe/markdown_0/imgs/img_in_image_box_897_291_953_346.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-05-15T14%3A48%3A07Z%2F-1%2F%2F133455374697e0c2e2978fb08551aeb40cc8fc846ca5564808a3c0e8f78349de" alt="Image" width="5%" /></div>


Hien Thi Nguyen $ ^{a} $, Hoang Nguyen $ ^{b} $, Minh-Ngoc Tran $ ^{c,*} $

 $ ^{a} $ Faculty of Mathematical Economics, Thuongmai University, Viet Nam

 $ ^{b} $ Department of Management and Engineering, Linköping University, Sweden

 $ ^{c} $ The University of Sydney Business School, Australia

### ARTICLE INFO

JEL classification:

C58

C53

C10

Keywords:

GARCH

GARCH-X

Volatility forecast

Realized measures

Sequence Monte Carlo

## ABSTRACT

Exogenous information such as policy news and economic indicators can have the potential to trigger significant movements in financial asset volatility. This article presents a model, called the RECH-X model, that allows incorporating exogenous variables into a recurrent neural network for volatility modeling and forecasting. The RECH-X model can allow for abrupt changes in the volatility level and effectively capture the complex serial dependence structure in the volatility dynamics. We demonstrate in a wide range of applications that the RECH-X model consistently outperforms the benchmark models in terms of volatility modeling and forecasting.

## 1. Introduction

Prediction of asset volatility plays a vital role in the financial services industry, including but not limited to risk management, capital allocation, and financial engineering. Empirical studies suggest that the volatility of financial assets exhibits highly complex behavior including, for example, clustering (Mandelbrot, 1967), asymmetric volatility response (Black, 1976), and leptokurtosis (Carnero et al., 2004). These phenomena reflect the long-term autocorrelations and non-linear sequential dependencies, which make modeling and predicting volatility a challenging endeavor. The seminal volatility models such as the Generalized Autoregressive Conditional Heteroscedasticity (GARCH) (Engle, 1982; Bollerslev, 1986) and the Stochastic Volatility (SV) (Taylor, 1982) have gained popularity among scholars and become industrial standards for their ability to capture some of the intertemporal volatility dynamics. However, these models have yet to capture the external information sources on asset volatility. The exogenous information sources can be realized volatility or economic news that signal the changes in the financial market condition. For example, realized volatility (RV) estimated using high-frequency data carries far more information about the current level of volatility than the daily squared returns (Andersen and Bollerslev, 1998; Andersen et al., 2001; Koopman et al., 2005; Hansen et al., 2012, 2014). Additionally, including economic news rather than just using the return time series itself in traditional econometric volatility models can result in increased predictive power (Engle and Rangel, 2008; Engle et al., 2013; Asgharian et al., 2013; Conrad and Kleen, 2020).

This paper introduces a new model, Recurrent Conditional Heteroscedasticity with exogenous variables, denoted as RECH-X, which builds upon the framework developed by Nguyen et al. (2022) and allows for the incorporation of multiple sources of exogenous data. In RECH-X, the long-term component of the volatility is governed by a recurrent neural network of exogenous variables to capture complex movements such as non-linearity and long-term dependence while the short-term component follows a GARCH-type process. Hence, the RECH-X model provides a flexible yet simple framework for incorporating a variety of exogenous

data to improve the modeling and forecasting of volatility. Our work also contributes to the growing collection of evidence that it is possible to harness the predictive power of deep learning models in financial risk modeling while managing some of their shortcomings through the use of hybrid deep learning econometric models.

In many non-financial applications, such as language translation or speech synthesis, where sequential data exhibit long-range memory and complex dynamics, deep learning has demonstrated its impressive predictive performance. Recurrent Neural Networks (RNN) are the current state-of-the-art deep learning models for solving sequential learning problems. RNN can effectively process sequential data by retaining memory of the past inputs and capturing non-linear temporal dependencies within the data (Lipton et al., 2015; Goodfellow et al., 2016). The application of deep learning in financial econometrics has also emerged recently. Roh (2007) uses a feed-forward neural network (FNN) as another processing layer; i.e., an FNN is used to model the volatility estimates produced by an econometric model and produce the final estimate of the volatility. Kim and Won (2018) extend this idea by using an RNN rather than an FNN to depict more accurately the temporal effects. Hajizadeh et al. (2012) and Kristjanpoller et al. (2014) propose alternative approaches to combining FNNs and EGARCH, demonstrating improved predictive power. Luo et al. (2021) forecast the realized volatility of crude oil futures prices based on machine learning; see, also Luo et al. (2022). These hybrid models, which combine econometric models with deep learning, are in general superior to the former in terms of predictive performance. However, they suffer from an important shortcoming: unlike traditional econometric models, these hybrid models can be opaque and difficult to interpret and to audit. To harness the predictive power of RNN while retaining the interpretability of traditional econometric models, Nguyen et al. (2022) propose a hybrid RNN and GARCH-type model, called the Recurrent Conditional Heteroscedasticity (RECH) model. Nguyen et al. (2022) demonstrate using simulations and empirical studies on several stock market indices that their hybrid model exhibits superior predictive properties relative to the traditional econometric models while remaining transparent and interpretable.

Adding exogenous sources of information into a traditional econometric model can lead to improved predictive performance, but it may require ad-hoc modifications, such as the need of an additional measurement equation to model the dynamics of the noisy. The Multiplicative error model (MEM) (Engle, 2002) marked a significant milestone in the integration of realized measure as the covariate within the GARCH-X model framework. Barndorff-Nielsen and Shephard (2002) further refined the specification by incorporating both realized variance and bi-power variation into their model. Engle and Gallo (2006) also contributed to this line of research by leveraging intra-daily data as a covariate. In subsequent developments, Shephard and Sheppard (2010) introduced the High-frequency-based volatility (HEAVY) model, while Hansen et al. (2012) presented the Realized GARCH model. These models both specify the conditional variance within the GARCH-X framework.

In addition, the incorporation of exogenous economic variables has been observed in several studies. Notably, Glosten et al. (1993a), Gray (1996) and Engle and Patton (2001) employed interest rate levels as covariates within the GARCH-X model. Similarly, Hodrick (1989) and Hagiwara and Herce (1999) incorporated forward-spot spreads and interest rate spreads between countries as covariates in their analysis. Following, Han and Park (2008) used the yield spread between Aaa and Baa bonds as the covariate in the GARCH-X model. Other examples of exogenous economic variables include gold price (Smith, 2001), interest rate volatility (Brenner et al., 1996; Gray, 1996), exchange rate (Bollerslev and Melvin, 1994).

By extracting the related information from exogenous variables, we demonstrate the flexibility and efficiency of the RECH-X model in forecasting through a range of examples. In the first application, we show that the RECH-X model using realized volatility as an exogenous variable outperforms the benchmark models such as GARCH and RealGARCH for the index returns of ten major stock markets in the world. The second application employs four financial-economic indicators, including gold prices, oil prices, exchange rates, and the US stock market uncertainty, for improving volatility forecasting of five financial markets. Our results show that the RECH-X model consistently predicts volatility better than the GARCH model. The US stock market uncertainty has a strong positive effect, while the effects of gold prices, oil prices, and exchange rates are somewhat diverse.

This paper is organized as follows. The benchmark models GARCH and RealGARCH, together with RECH and its extension RECH-X, are presented in Section 2. Bayesian inference is presented in Section 3. Empirical results are presented in Sections 4 and 5 concludes. The computer code is available upon contacting the authors.

## 2. Volatility models

This section first discusses the GARCH(1,1) model of Bollerslev (1986), with Student-t innovations. There exist a vast number of GARCH-type models, however, Hansen and Lunde (2005) find no evidence of significant outperformance over GARCH. We therefore use GARCH as the benchmark model in this paper. We also describe the RealGARCH model of Hansen et al. (2012) when realized measures are available. We then describe the RECH model of Nguyen et al. (2022), and discuss how to incorporate exogenous information into RECH to improve its volatility modeling and forecasting.

### 2.1. GARCH, GARCH-X and RealGARCH

Let $\{y_t, t = 1, \ldots, T\}$ be the daily return time series and $\{RV_t, t = 1, \ldots, T\}$ the corresponding realized volatilities. We consider the GARCH model with Student's $t$ innovations:

 $$ y_{t}=\sigma_{t}\epsilon_{t},\ \epsilon_{t}\stackrel{i i d}{\sim}t_{v},\ t=1,2,\ldots,T, $$ 

 $$ \sigma_{t}^{2}=\omega+\alpha y_{t-1}^{2}+\beta\sigma_{t-1}^{2},\ t=2,\ldots,T. $$ 

As typical in the literature, we impose the condition that  $ \alpha > 0 $,  $ \beta > 0 $ and  $ \alpha \frac{\nu}{\nu - 2} + \beta < 1 $ for the stationarity of  $ y_t $. Also, we restrict  $ \nu > 2 $ to ensure a finite variance of the  $ y_t $. We set  $ \sigma_1^2 $ to be the sample variance of the training data following the common practice in the literature. A straightforward extension of the GARCH model is a GARCH-X which incorporates related exogenous information for volatility modeling:

 $$ \begin{aligned}&y_{t}=\sigma_{t}\epsilon_{t},\ \epsilon_{t}\stackrel{i i d}{\sim}t_{v},\ t=1,2,\ldots,T,\\&\sigma_{t}^{2}=\omega+\alpha y_{t-1}^{2}+\beta\sigma_{t-1}^{2}+\pi^{\top}x_{t-1},\ t=2,\ldots,T.\\ \end{aligned} $$ 

Here,  $ x_t $ is the vector of non-negative exogenous variables and  $ \pi $ the vector of non-negative coefficients. For negative  $ x_t $, one often replaces  $ \pi^\top x_{t-1} $ with  $ \pi^\top x_{t-1}^2 $ (Francq and Thieu, 2019).

When high-frequency data are available, one of the most successful econometric models that incorporate RV into volatility modeling is perhaps the non-exponential RealGARCH model of (Hansen et al., 2012):

 $$ y_{t}=\sigma_{t}\epsilon_{t},\ \epsilon_{t}\stackrel{i i d}{\sim}t_{v},\ t=1,2,\ldots,T, $$ 

 $$ \sigma_{t}^{2}=\omega+\beta\sigma_{t-1}^{2}+\gamma\mathrm{R V}_{t-1},\ t=2,\ldots,T, $$ 

 $$ \mathrm{RV}_{t}=\xi+\varphi\sigma_{t}^{2}+\tau_{1}\epsilon_{t}+\tau_{2}\Big(\frac{\nu-2}{\nu}\epsilon_{t}^{2}-1\Big)+u_{t},\ t=1,\ldots,T,\ u_{t}\stackrel{i i d}{\sim}\mathcal{N}(0,\sigma_{u}^{2}). $$ 

The model parameters include  $ \nu $,  $ \omega $,  $ \beta $,  $ \gamma $,  $ \xi $,  $ \varphi $,  $ \tau_1 $,  $ \tau_2 $ and  $ \sigma_u^2 $. To ensure that  $ \sigma_t^2 $ is finite and positive, we impose the following conditions:  $ \omega + \gamma \xi > 0 $ and  $ 0 < \beta + \gamma \varphi < 1 $; see Hansen et al. (2012) and Gerlach and Wang (2016) for more details. The non-exponential RealGARCH of Hansen et al. (2012) can be considered as an extension of the GARCH(0,1)-X model (Han, 2015).

This paper employs the GARCH, the RealGARCH (when RV is available) and the GARCH-X as the benchmark models to assess the performance of the RECH-X model described in the next section.

### 2.2. Recurrent conditional heteroscedasticity

The RECH models of Nguyen et al. (2022) are a class of volatility models that incorporate an RNN within a GARCH-type model for flexible volatility modeling. Its key motivation is to use an additive component governed by an RNN to capture the complex serial dependence structure in the volatility dynamics which might be overlooked by the GARCH component. The general RECH model is:

 $$ y_{t}=\sigma_{t}\epsilon_{t},\quad\epsilon_{t}\sim i.i.d,\quad t=1,2,\ldots,T, $$ 

 $$ \sigma_{t}^{2}=g(\omega_{t})+f(\sigma_{t-1}^{2},\ldots,\sigma_{t-p}^{2},y_{t-1},\ldots,y_{t-q}),\ t=q+1,\ldots,T, $$ 

 $$ \omega_{t}=\beta_{0}+\beta_{1}h_{t},\ t=1,\ldots,T, $$ 

 $$ h_{t}=\mathrm{RNN}(x_{t},h_{t-1}),\ t=2,\ldots,T,\ \mathrm{with}\ h_{1}\equiv0. $$ 

Here,  $ f(\sigma_{t-1}^2, \ldots, \sigma_{t-p}^2, y_{t-1}, \ldots, y_{t-q}) $ is the GARCH-type component with  $ p $ and  $ q $ the lag orders of  $ \sigma_t^2 $ and  $ y_t $ respectively. The function  $ g(\omega_t) $ is called the recurrent component with  $ \omega_t $ governed by an RNN, where  $ g(\cdot) $ is an activation function;  $ \beta_0 $ and  $ \beta_1 $ are the coefficients. The input vector  $ x_t $ in the calculation of the recurrent state  $ h_t = \text{RNN}(x_t, h_{t-1}) $ is a vector of additional information sources whose choice is discussed shortly. The RECH model reduces to the GARCH model if  $ \beta_1 = 0 $ and  $ g(\omega_t) = \omega_t $.

The recurrent state  $ h_{t} $ in (1d) can be modeled by any RNN framework. Nguyen et al. (2022) use a simple RNN (see Eq. (2d)), and Liu et al. (2023) opt for the long-short term memory (LSTM) model of Hochreiter and Schmidhuber (1997) as an alternative. The LSTM uses gating mechanisms to control the flow of information through the network and enable the RNN to selectively retain and forget information over long time horizons. Liu et al. (2023) demonstrate that using the LSTM in general improves the predictive performance of RECH. However, the simple RNN has a simpler structure that allows for the interpretation of the influencing variables and requires less computational cost compared to LSTM. This paper focuses on testing the ability of the RECH model when incorporating exogenous information. To avoid possible confounding results when a sophisticated LSTM architecture is used, and to increase the interpretation, we opt for the simple RNN rather than LSTM. We consider GARCH(1,1) for the GARCH component and Student's t innovations. The model is fully written as follows:

 $$ y_{t}=\sigma_{t}\epsilon_{t},\ \epsilon_{t}\stackrel{i i d}{\sim}t_{v},\ t=1,2,\ldots,T, $$ 

 $$ \sigma_{t}^{2}=\omega_{t}+\alpha y_{t-1}^{2}+\beta\sigma_{t-1}^{2},\ t=2,\ldots,T, $$ 

 $$ \omega_{t}=\beta_{0}+\beta_{1}h_{t},\ t=1,\ldots,T, $$ 

 $$ h_{t}=\Psi(v^{\top}x_{t}+w_{h}h_{t-1}+b),\ t=2,\ldots,T,\ with\ h_{1}\equiv0. $$ 

Here,  $ \Psi(\cdot) $ is a non-linear activation function, which is chosen to be the ReLU  $ \Psi(x) = \max\{x, 0\} $. The GARCH parameters  $ \alpha $ and  $ \beta $ are imposed the usual constraint that  $ \alpha, \beta > 0 $ and  $ \alpha\frac{v}{v-2} + \beta < 1 $, which guarantees that the volatility  $ \sigma_t^2 $ is positive and finite for all  $ t $; see Nguyen et al. (2022). We set the initial  $ \sigma_t^2 $ as before. The input vector  $ x_t = (\omega_{t-1}, y_{t-1}, \sigma_{t-1}^2) $ in Nguyen et al. (2022) contains the past information of the level of long-term volatility, return, and total volatility. However, it is easy to incorporate exogenous variables into its recurrent component whenever such inputs are available and useful in terms of modeling and predicting the distribution of  $ y_t $. Hence, with the availability of an exogenous variable  $ z_{t-1} $ at time  $ t $, we propose to use  $ x_t = (\omega_{t-1}, y_{t-1}, \sigma_{t-1}^2, z_{t-1})^T $ and refer to the

resulting model in (2a)–(2d) as RECH-X. For example, with the choice of  $ z_{t-1} $ as  $ RV_{t-1} $, the vector  $ v $ in (2d) has four coefficients and we denote by  $ v_{\omega}, v_{\nu}, v_{\sigma}, v_{\mathrm{RV}} $ the coefficients with respect to  $ \omega_{t-1}, y_{t-1}, \sigma_{t-1}^{2}, RV_{t-1} $, respectively. The magnitude of  $ v_{\mathrm{RV}} $ characterizes the importance of realized measures for modeling the daily volatility  $ \sigma_{t}^{2} $. We demonstrate in Section 4.1 that  $ v_{\mathrm{RV}} $ is always significant across all the datasets considered. When multiple exogenous variables are used as in Section 4.2,  $ z_{t} $ and its corresponding coefficients  $ v_{z} $ become vectors. The flexible structure of RECH-X allows it to be able to capture the non-linear and long-term effects that the exogenous variables have on the volatility.

When high-frequency data are available, Liu et al. (2023) extend the RealGARCH model by incorporating an RNN, similar to the approach used in RECH, and name their model RealRECH. The RECH-X model differs from the RealRECH model in a significant way. Specifically, RECH-X adopts a distinct strategy by directly incorporating the exogenous information into the volatility equation, rather than introducing a measurement equation for the realized volatility as in RealRECH. This modification is made to accommodate the inclusion of other exogenous variables, which may not be feasible to include in the measurement equation. As a result, RECH-X offers greater flexibility in allowing market participants to model the volatility by making use of timely available related variables.

## 3. Bayesian inference

We adopt the Bayesian approach for inference and prediction in this paper. We follow Nguyen et al. (2022) and use their priors for the RECH-X parameters. In particular, the priors of the parameters in the GARCH component are  $ \alpha \sim U(0,1) $ and  $ \beta \sim U(0,1) $; the priors of the coefficients  $ \beta_0 $ and  $ \beta_1 $ are  $ \beta_0 \sim U(0,0.5) $ and  $ \beta_1 \sim U(0,0.5) $; and we use a normal prior with a zero mean and variance 0.1 for the recurrence parameters of the RNN component,  $ v_{\omega}, v_y, v_{\sigma}, w_h, b $, as they are often small. The prior for the regressors coefficients  $ v_z $ in (2d) is  $ N(0,0.5) $. Lastly, the prior of the degrees of freedom is  $ \nu \sim \text{Gamma}(1,0.1) $. For the parameters of the GARCH and the RealGARCH model, we employ the commonly used priors in the literature: see. e.g., Gerlach and Wang (2016).

Due to the complex nature of the RECH-X model and its challenging posterior distribution, we use the likelihood-annealing SMC method (Duan and Fulop, 2015) to sample from the posterior. Besides, the likelihood-annealing SMC also provides an estimation of the log marginal likelihood for the model comparison.

Let  $ y = \{y_t, t = 1, \ldots, T\} $ be the training data and  $ \theta $ the set of model parameters. The likelihood-annealing SMC samples sequentially from the prior distribution  $ p(\theta) $ to the posterior distribution  $ p(\theta|y) \propto p(y|\theta)p(\theta) $ through a sequence of annealed distributions,

 $$ \pi_{k}(\theta)\propto p(\theta)p(y|\theta)^{a_{k}},\ k=1,\ldots,K $$ 

with 0 = a1 < a2 < … < aK = 1. The samples drawn from the last annealed distribution πK(θ) are from the posterior p(θ|y).

For expanding-window out-of-sample prediction that updates the posterior each time a new observation arrives, we employ the data-annealing SMC method that samples from the sequence

 $$ \pi_{t}(\theta)\propto p(\theta)p(y_{1},\ldots,y_{T+t}|\theta),\ t=1,2,\ldots $$ 

We refer the reader to Nguyen et al. (2022) for the detailed implementation of the likelihood-annealing and data-annealing SMC samplers for RECH, and to Gunawan et al. (2022) for a general discussion of these SMC samplers.

## 4. Empirical illustrations

This section demonstrates the performance of the RECH-X model in comparison with the benchmark models using stock index and exogenous data from various countries. The first example tests the performance of RECH-X using a realized measure as the exogenous data. We use daily index returns from ten major stock markets including the S&P500 index and the Japanese Nikkei 225 index. The data are obtained from the Oxford-Man Institute. Table 9 in Appendix B provides the detailed description of the data. Each series consists of 2001 closed prices, leading to 2000 return values where the first 1500 observations were used for training and the rest for out-of-sample testing. In order to assess the predictive performance, we compare the one-step-ahead point and density forecasts of the last 500 observations using five predictive scores: the partial predictive score (PPS), the quantile score (QS), the mean squared forecast error (MSE), the mean absolute forecast error (MAE) and the R2LOG; see the definition in Appendix A.

The second example evaluates the performance of the RECH-X model using multiple financial-economic indicators as exogenous variables. The motivation is that the volatility of the stock yield series is not solely determined by past shocks and fluctuations but also impacted by macroeconomic factors. According to the studies conducted by Bekhet and Mugableh (2012), Khan (2014) and Singhal et al. (2019) there is a positive correlation between GDP and stock prices. Conversely, interest rates are believed to have a negative relationship with stock returns. Research by Humpe and Macmillan (2007), Abugri (2008) and Alam and Uddin (2009) demonstrate that interest rates negatively impact on share prices as well as portfolio returns. Kasman (2003) shows that exchange rates have a consistently stable relationship with stock indices. Furthermore, empirical findings by Kandir (2008) and Aydemir and Demirhan (2009) suggest that exchange rates have an impact on portfolio returns. Various publications have shown that inflation and money supply have an impact on stock market fluctuations. Maysami et al. (2004), by studying the Singapore market and Adam and Tweneboah (2008), focusing on the Ghanaian market, all find a significant positive relationship between inflation (CPI) and stock returns. Additionally, Le et al. (2019) discover that the stock price index has a causal relationship with the M2 money supply, specifically in the short-term. Oil prices and gold prices are considered indicators that potentially influence the stock market index. According to Smyth and Narayan (2018), an upsurge in oil prices can result in a decline in future dividend stream profits. Mokni

<div style="text-align: center;"><div style="text-align: center;">Table 1 S&P500 data: Posterior means, with the posterior standard deviations in brackets, of the main model parameters. The last column shows the log marginal likelihood estimates with the Monte Carlo standard errors in brackets, across 10 different runs of the likelihood-annealing SMC sampler. The number in bold indicates the best value.</div> </div>




<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>Model</td><td style='text-align: center; word-wrap: break-word;'>$ \alpha/\gamma $</td><td style='text-align: center; word-wrap: break-word;'>$ \beta $</td><td style='text-align: center; word-wrap: break-word;'>v</td><td style='text-align: center; word-wrap: break-word;'>$ \varphi/\beta_{1} $</td><td style='text-align: center; word-wrap: break-word;'>v_{RV}</td><td style='text-align: center; word-wrap: break-word;'>Mar.llh</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>GARCH</td><td style='text-align: center; word-wrap: break-word;'>0.118 (0.021)</td><td style='text-align: center; word-wrap: break-word;'>0.764 (0.037)</td><td style='text-align: center; word-wrap: break-word;'>4.892 (0.644)</td><td style='text-align: center; word-wrap: break-word;'></td><td style='text-align: center; word-wrap: break-word;'></td><td style='text-align: center; word-wrap: break-word;'>-1556.3 (0.21)</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>RealGARCH</td><td style='text-align: center; word-wrap: break-word;'>0.001 (0.000)</td><td style='text-align: center; word-wrap: break-word;'>0.997 (0.000)</td><td style='text-align: center; word-wrap: break-word;'>3.376 (0.316)</td><td style='text-align: center; word-wrap: break-word;'>1.281 (0.204)</td><td style='text-align: center; word-wrap: break-word;'></td><td style='text-align: center; word-wrap: break-word;'>-1595.0 (0.18)</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>RECH-X</td><td style='text-align: center; word-wrap: break-word;'>0.018 (0.013)</td><td style='text-align: center; word-wrap: break-word;'>0.185 (0.164)</td><td style='text-align: center; word-wrap: break-word;'>6.594 (1.000)</td><td style='text-align: center; word-wrap: break-word;'>0.375 (0.122)</td><td style='text-align: center; word-wrap: break-word;'>0.412 (0.158)</td><td style='text-align: center; word-wrap: break-word;'>-1487.4 (0.19)</td></tr></table>

<div style="text-align: center;"><div style="text-align: center;">Table 2</div> </div>


<div style="text-align: center;"><div style="text-align: center;">S&P500 data: summary statistics of normalized residuals.</div> </div>




<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>Model</td><td style='text-align: center; word-wrap: break-word;'>Mean</td><td style='text-align: center; word-wrap: break-word;'>Std</td><td style='text-align: center; word-wrap: break-word;'>Skew</td><td style='text-align: center; word-wrap: break-word;'>Kurtosis</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>GARCH</td><td style='text-align: center; word-wrap: break-word;'>0.050</td><td style='text-align: center; word-wrap: break-word;'>0.995</td><td style='text-align: center; word-wrap: break-word;'>$ -0.242 $</td><td style='text-align: center; word-wrap: break-word;'>2.953</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>RealGARCH</td><td style='text-align: center; word-wrap: break-word;'>0.065</td><td style='text-align: center; word-wrap: break-word;'>1.009</td><td style='text-align: center; word-wrap: break-word;'>$ -0.172 $</td><td style='text-align: center; word-wrap: break-word;'>2.821</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>RECH-X</td><td style='text-align: center; word-wrap: break-word;'>0.044</td><td style='text-align: center; word-wrap: break-word;'>1.037</td><td style='text-align: center; word-wrap: break-word;'>$ -0.296 $</td><td style='text-align: center; word-wrap: break-word;'>2.943</td></tr></table>

and Youssef (2019) found a positive relationship between crude oil prices and stock markets in Gulf Cooperation Council (GCC) member countries. As for gold prices, Singhal et al. (2019) and Akbar et al. (2019) argue that fluctuations in gold prices negatively affect investment returns on the stock market.

The RECH-X model offers an improvement over the RECH model by incorporating covariates that represent the impact of economic and financial indicators on stock returns in the heteroskedasticity model. Therefore, we consider five stock markets including Vietnam (VN100), Japan (N225), France (CAC40), Australia (ASX) and Brazil (BVSP). The data were collected from the financial portal https://investing.com for the period from January 2015 to April 2023 that includes the Covid-19 period, with about 2000 observations. Each dataset is divided into two parts: the in-sample data comprises the first 1000 observations, while the remaining data is utilized for evaluating out-of-sample prediction performance. Data characteristics of economic and financial indicators have differences in publication frequency for example, gold prices, oil prices, exchange rates are announced at high frequency such as day or hour while inflation, interest rates, export growth are announced at monthly frequency, GDP quarterly frequency. However, in this study, we only consider and include in the RECH model exogenous variables with the same frequency as the return of stocks. This example examines the predictive ability of financial-economic indicators for volatility modeling and forecasting. Four indicators are considered in this paper: the currency exchange rates of the above countries against the USD, the uncertainty of the US stock market via the VIX index, Gold price and Crude oil price (West Texas Intermediate — WTI).

### 4.1. The effect of realized measures on volatility forecast

We discuss the results for the selected markets S&P500 and N225 in detail in this section, and the results for the other markets can be found in Appendix B.

#### 4.1.1. S&P500 data

The in-sample estimation results of the Standard & Poor's 500 index returns are summarized in Table 1; only the main interested parameters are shown. The posterior mean of the coefficient with respect to the realized volatility  $ v_{RV} $ is more than two standard deviations from zero, indicating the significant effect of the realized volatility on the volatility dynamic. The degrees of freedom in GARCH and RealGARCH models are smaller than those in the RECH-X model meaning that it is less likely to observe extreme innovations in the RECH-X model. The last column in Table 1 lists the log marginal likelihood estimates, which show that the S&P500 data strongly support the RECH-X model.

Table 2 shows the summary statistics of the normalized residuals for in-sample data, i.e.  $ \widehat{\varepsilon}_i = \Phi^{-1}(F_{i_v}(y_i/\widehat{\sigma}_i)) $, where  $ \Phi(\cdot) $ and  $ F_{i_v}(\cdot) $ are the cumulative probability distribution functions of the standard normal distribution and Student's t distribution  $ t_v $, respectively. These normalized residuals are expected to follow the standard normal distribution. As shown, for each of the three models, the residuals' standard deviation and kurtosis are close to 1 and 3 — the quantities of the standard normal distribution, respectively. However, all three models produce residuals that are still slightly skewed to the left; perhaps, the RECH specification that combines RNN with the GJR model (Glosten et al., 1993b) can increase the in-sample fit further.

Fig. 1 plots the return indices in the out-of-sample data together with 95% one-step-ahead forecast intervals from the three models. The forecast intervals generated by RECH-X can quickly expand to accommodate extreme observations and decay at a faster rate compared to the benchmark models. The predictive performance of the three models is summarized in Table 3, which shows that the RECH-X model has the best performance across all the five predictive scores. The RealGARCH model performs better the GARCH model in terms of MSE and MAE, but not PPS, QS and R2LOG.

<div style="text-align: center;"><div style="text-align: center;">Table 3 S&P500 data: Predictive performance. The numbers in bold indicate the best values.</div> </div>




<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>Model</td><td style='text-align: center; word-wrap: break-word;'>PPS</td><td style='text-align: center; word-wrap: break-word;'>QS</td><td style='text-align: center; word-wrap: break-word;'>MSE</td><td style='text-align: center; word-wrap: break-word;'>MAE</td><td style='text-align: center; word-wrap: break-word;'>R2LOG</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>GARCH</td><td style='text-align: center; word-wrap: break-word;'>1.225</td><td style='text-align: center; word-wrap: break-word;'>0.035</td><td style='text-align: center; word-wrap: break-word;'>0.138</td><td style='text-align: center; word-wrap: break-word;'>0.304</td><td style='text-align: center; word-wrap: break-word;'>1.079</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>RealGARCH</td><td style='text-align: center; word-wrap: break-word;'>1.236</td><td style='text-align: center; word-wrap: break-word;'>0.037</td><td style='text-align: center; word-wrap: break-word;'>0.120</td><td style='text-align: center; word-wrap: break-word;'>0.283</td><td style='text-align: center; word-wrap: break-word;'>1.129</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>RECH-X</td><td style='text-align: center; word-wrap: break-word;'>1.139</td><td style='text-align: center; word-wrap: break-word;'>0.032</td><td style='text-align: center; word-wrap: break-word;'>0.095</td><td style='text-align: center; word-wrap: break-word;'>0.234</td><td style='text-align: center; word-wrap: break-word;'>0.670</td></tr></table>

<div style="text-align: center;"><div style="text-align: center;">Out-of-sample analysis of SP500</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//cc1fdb5a-1eb5-4e51-afcc-5921f2398111/markdown_0/imgs/img_in_chart_box_108_276_978_727.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-05-15T14%3A48%3A42Z%2F-1%2F%2F658482d42956d3c0f9b6af27331daf239eb0a52713bc81d25ea1c5e3f5a3a7a6" alt="Image" width="79%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Fig. 1. S&P500: plots of return indices in the out-of-sample data together with 95% one-step-ahead forecast intervals from GARCH (dashed green), RealGARCH (dash-dotted blue) and RECH-X (solid black).</div> </div>


<div style="text-align: center;"><div style="text-align: center;">N225 data: Posterior means, with the posterior standard deviations in brackets, of the main model parameters. The last column shows the log marginal likelihood estimates with the Monte Carlo standard errors in brackets, across 10 different runs of the SMC sampler. The number in bold indicates the best value.</div> </div>




<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>Model</td><td style='text-align: center; word-wrap: break-word;'>$ \alpha/\gamma $</td><td style='text-align: center; word-wrap: break-word;'>$ \beta $</td><td style='text-align: center; word-wrap: break-word;'>v</td><td style='text-align: center; word-wrap: break-word;'>$ \varphi/\beta_{1} $</td><td style='text-align: center; word-wrap: break-word;'>v_{RV}</td><td style='text-align: center; word-wrap: break-word;'>Mar.llh</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>GARCH</td><td style='text-align: center; word-wrap: break-word;'>0.072 (0.015)</td><td style='text-align: center; word-wrap: break-word;'>0.872 (0.026)</td><td style='text-align: center; word-wrap: break-word;'>5.494 (0.783)</td><td style='text-align: center; word-wrap: break-word;'></td><td style='text-align: center; word-wrap: break-word;'></td><td style='text-align: center; word-wrap: break-word;'>-2399.8 (0.18)</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>RealGARCH</td><td style='text-align: center; word-wrap: break-word;'>0.538 (0.069)</td><td style='text-align: center; word-wrap: break-word;'>0.368 (0.055)</td><td style='text-align: center; word-wrap: break-word;'>5.567 (0.780)</td><td style='text-align: center; word-wrap: break-word;'>0.631 (0.065)</td><td style='text-align: center; word-wrap: break-word;'></td><td style='text-align: center; word-wrap: break-word;'>-2382.6 (0.23)</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>RECH-X</td><td style='text-align: center; word-wrap: break-word;'>0.031 (0.014)</td><td style='text-align: center; word-wrap: break-word;'>0.497 (0.211)</td><td style='text-align: center; word-wrap: break-word;'>5.687 (0.772)</td><td style='text-align: center; word-wrap: break-word;'>0.399 (0.114)</td><td style='text-align: center; word-wrap: break-word;'>0.617 (0.186)</td><td style='text-align: center; word-wrap: break-word;'>-2371.0 (0.20)</td></tr></table>

#### 4.1.2. N225 data

The analysis of the Japanese Nikkei 225 index returns shows similar results to those of the S&P500 data. The in-sample estimation results are summarized in Table 4. Even that the degrees of freedom are similar among the models, the posterior mean of  $ \nu_{RV} $ is still more than two standard deviations from zero, indicating the significant forecast ability of the realized volatility on the underlying volatility dynamic. The last column lists the log marginal likelihood estimates, which shows that the data strongly support the RECH-X model.

Table 5 summarizes the predictive performance of the three models, which, similar to the case with the S&P500 data, shows that the RECH-X model has the best performance across all the five predictive scores. Fig. 2 plots of the return indices in the testing data together with 95% one-step-ahead forecast intervals from the three models. As shown, the RECH-X forecasts trace the data well and be responsive to abrupt changes in the returns.

We also examine eight other stock indices; detailed results can be found in Appendix B. These results consistently confirm the superiority of RECH-X over the benchmark models GARCH and RealGARCH.

<div style="text-align: center;"><div style="text-align: center;">Table 5</div> </div>


<div style="text-align: center;"><div style="text-align: center;">N225 data: Predictive performance. The numbers in bold indicate the best values.</div> </div>




<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>Model</td><td style='text-align: center; word-wrap: break-word;'>PPS</td><td style='text-align: center; word-wrap: break-word;'>QS</td><td style='text-align: center; word-wrap: break-word;'>MSE</td><td style='text-align: center; word-wrap: break-word;'>MAE</td><td style='text-align: center; word-wrap: break-word;'>R2LOG</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>GARCH</td><td style='text-align: center; word-wrap: break-word;'>1.459</td><td style='text-align: center; word-wrap: break-word;'>0.041</td><td style='text-align: center; word-wrap: break-word;'>0.324</td><td style='text-align: center; word-wrap: break-word;'>0.520</td><td style='text-align: center; word-wrap: break-word;'>2.170</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>RealGARCH</td><td style='text-align: center; word-wrap: break-word;'>1.454</td><td style='text-align: center; word-wrap: break-word;'>0.040</td><td style='text-align: center; word-wrap: break-word;'>0.321</td><td style='text-align: center; word-wrap: break-word;'>0.537</td><td style='text-align: center; word-wrap: break-word;'>2.321</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>RECH-X</td><td style='text-align: center; word-wrap: break-word;'>1.416</td><td style='text-align: center; word-wrap: break-word;'>0.037</td><td style='text-align: center; word-wrap: break-word;'>0.317</td><td style='text-align: center; word-wrap: break-word;'>0.501</td><td style='text-align: center; word-wrap: break-word;'>1.933</td></tr></table>

<div style="text-align: center;"><div style="text-align: center;">Out-of-sample analysis of N225</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//cc1fdb5a-1eb5-4e51-afcc-5921f2398111/markdown_1/imgs/img_in_chart_box_107_265_978_716.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-05-15T14%3A48%3A43Z%2F-1%2F%2F12854a5cba6acddee5a0c53949186391f4f6a851731ad614f77906028a763665" alt="Image" width="79%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Fig. 2. N225: plots of return indices in the testing data together with 95% one-step-ahead forecast intervals from GARCH (dashed green), RealGARCH (dashed-dotted blue) and RECH-X (solid black).</div> </div>


### 4.2. The effect of financial-economic indicators on volatility forecast

Scholars have shown that key financial-economic indicators such as economic growth, exchange rates, gold prices, oil prices, interest rates, inflation and money supply, might have an impact on stock market volatility. Furthermore, fluctuations of major stock markets in the world can also affect the stock market of other countries due to the spillover effect. This study focuses on examining how exchange rates, gold prices, oil prices, and the uncertainty of the US stock market influence the volatility of five stock markets: Vietnam (VN100), Japan (N255), France (CAC40), Australia (ASX), and Brazil (BVSP). Other macroeconomic indicators such as GDP can also be considered. We use these four variables as the inputs  $ z_{t} $ of the RECH-X model. It is important to note that, as the input variables to the RNN, these exogenous variables should be standardized. More precisely, for each input time series, the in-sample data are standardized to have a zero mean and standard deviation of 1, and the out-of-sample data are standardized accordingly using the mean and standard deviation computed from the in-sample data.

Exchange rates: There is mixed evidence about the effect of exchange rates on the volatility of stock markets. Tursoy et al. (2008) and Abugri (2008) find that the exchange rate does not affect stock market volatility. On the other hand, Adam and Tweneboah (2008) obtain a negative effect while Maysami et al. (2004) support the hypothesis of a positive relationship between exchange rates and stock returns.

Gold: Gold and stocks are two alternative options that investors often consider. Gold is commonly regarded as a safe haven asset that offers financial stability during economic downturns, while stocks present the potential for higher long-term returns but also come with higher risk and uncertainty. Smith (2001) demonstrates bi-directional causality between gold and stock prices using the Granger test. Garefalakis et al. (2011) argue that the volatility of gold prices negatively affects investment returns on the Hong Kong stock market. Akbar et al. (2019) employ a vector autoregressive model to illustrate a reverse effect between gold price and stock price, while Singhal et al. (2019) show that the world gold price has a positive influence on Mexico's stock price. These studies provide valuable insights into the complex relationship between gold and stocks in various markets.

Crude oil: Empirical results on the relationship between oil and stock prices have been shown to be positive or negative depending on the stock markets considered. Singhal et al. (2019) show that oil prices negatively affect stock prices of Mexico, whereas Mokni and Youssef (2019) and Khan et al. (2021) show a positive relationship between crude oil price and stock market.

The uncertainty of the US stock market: The VIX is a popular measure of the US stock market volatility expectations based on S&P500 index options. Due to the interconnectedness of global financial markets, it is generally expected that the volatility of

<div style="text-align: center;"><div style="text-align: center;">Data from January 2015 to April 2023: Posterior means, with the posterior standard deviations in brackets, of the main model parameters. The last column shows the log marginal likelihood, the number in bold indicates the best value.</div> </div>




<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>Data</td><td style='text-align: center; word-wrap: break-word;'>Model</td><td style='text-align: center; word-wrap: break-word;'>w</td><td style='text-align: center; word-wrap: break-word;'>$ \alpha $</td><td style='text-align: center; word-wrap: break-word;'>$ \beta $</td><td style='text-align: center; word-wrap: break-word;'>$ \nu $</td><td style='text-align: center; word-wrap: break-word;'>$ \beta_{1} $</td><td style='text-align: center; word-wrap: break-word;'>Mar.llh</td></tr><tr><td rowspan="3">VN100</td><td style='text-align: center; word-wrap: break-word;'>GARCH</td><td style='text-align: center; word-wrap: break-word;'>0.033 (0.013)</td><td style='text-align: center; word-wrap: break-word;'>0.078 (0.019)</td><td style='text-align: center; word-wrap: break-word;'>0.83 (0.041)</td><td style='text-align: center; word-wrap: break-word;'>5.53 (0.818)</td><td style='text-align: center; word-wrap: break-word;'></td><td style='text-align: center; word-wrap: break-word;'>-1308.5 (0.094)</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>GARCH-X</td><td style='text-align: center; word-wrap: break-word;'>0.083 (0.041)</td><td style='text-align: center; word-wrap: break-word;'>0.092 (0.025)</td><td style='text-align: center; word-wrap: break-word;'>0.585 (0.141)</td><td style='text-align: center; word-wrap: break-word;'>5.547 (0.843)</td><td style='text-align: center; word-wrap: break-word;'></td><td style='text-align: center; word-wrap: break-word;'>-1322.2 (0.785)</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>RECH-X</td><td style='text-align: center; word-wrap: break-word;'>0.162 (0.284)</td><td style='text-align: center; word-wrap: break-word;'>0.064 (0.022)</td><td style='text-align: center; word-wrap: break-word;'>0.708 (0.085)</td><td style='text-align: center; word-wrap: break-word;'>6.156 (1.007)</td><td style='text-align: center; word-wrap: break-word;'>0.169 (0.073)</td><td style='text-align: center; word-wrap: break-word;'>-1303.5 (0.36)</td></tr><tr><td rowspan="3">CAC40</td><td style='text-align: center; word-wrap: break-word;'>GARCH</td><td style='text-align: center; word-wrap: break-word;'>0.022 (0.012)</td><td style='text-align: center; word-wrap: break-word;'>0.089 (0.023)</td><td style='text-align: center; word-wrap: break-word;'>0.841 (0.041)</td><td style='text-align: center; word-wrap: break-word;'>5.199 (0.830)</td><td style='text-align: center; word-wrap: break-word;'></td><td style='text-align: center; word-wrap: break-word;'>-1413.1 (0.083)</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>GARCH-X</td><td style='text-align: center; word-wrap: break-word;'>0.032 (0.019)</td><td style='text-align: center; word-wrap: break-word;'>0.117 (0.030)</td><td style='text-align: center; word-wrap: break-word;'>0.762 (0.065)</td><td style='text-align: center; word-wrap: break-word;'>5.342 (0.820)</td><td style='text-align: center; word-wrap: break-word;'></td><td style='text-align: center; word-wrap: break-word;'>-1431.0 (0.289)</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>RECH-X</td><td style='text-align: center; word-wrap: break-word;'>0.055 (0.243)</td><td style='text-align: center; word-wrap: break-word;'>0.045 (0.020)</td><td style='text-align: center; word-wrap: break-word;'>0.616 (0.116)</td><td style='text-align: center; word-wrap: break-word;'>5.705 (0.808)</td><td style='text-align: center; word-wrap: break-word;'>0.313 (0.092)</td><td style='text-align: center; word-wrap: break-word;'>-1388.7 (0.27)</td></tr><tr><td rowspan="3">N225</td><td style='text-align: center; word-wrap: break-word;'>GARCH</td><td style='text-align: center; word-wrap: break-word;'>0.052 (0.021)</td><td style='text-align: center; word-wrap: break-word;'>0.099 (0.024)</td><td style='text-align: center; word-wrap: break-word;'>0.773 (0.050)</td><td style='text-align: center; word-wrap: break-word;'>4.034 (0.523)</td><td style='text-align: center; word-wrap: break-word;'></td><td style='text-align: center; word-wrap: break-word;'>-1538.2 (0.528)</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>GARCH-X</td><td style='text-align: center; word-wrap: break-word;'>0.031 (0.019)</td><td style='text-align: center; word-wrap: break-word;'>0.105 (0.028)</td><td style='text-align: center; word-wrap: break-word;'>0.743 (0.063)</td><td style='text-align: center; word-wrap: break-word;'>4.087 (0.542)</td><td style='text-align: center; word-wrap: break-word;'></td><td style='text-align: center; word-wrap: break-word;'>-1524.6 (1.156)</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>RECH-X</td><td style='text-align: center; word-wrap: break-word;'>0.245 (0.207)</td><td style='text-align: center; word-wrap: break-word;'>0.029 (0.018)</td><td style='text-align: center; word-wrap: break-word;'>0.583 (0.095)</td><td style='text-align: center; word-wrap: break-word;'>4.962 (0.712)</td><td style='text-align: center; word-wrap: break-word;'>0.406 (0.107)</td><td style='text-align: center; word-wrap: break-word;'>-1483.5 (0.403)</td></tr><tr><td rowspan="3">ASX</td><td style='text-align: center; word-wrap: break-word;'>GARCH</td><td style='text-align: center; word-wrap: break-word;'>0.778 (0.128)</td><td style='text-align: center; word-wrap: break-word;'>0.234 (0.045)</td><td style='text-align: center; word-wrap: break-word;'>0.238 (0.076)</td><td style='text-align: center; word-wrap: break-word;'>7.53 (1.378)</td><td style='text-align: center; word-wrap: break-word;'></td><td style='text-align: center; word-wrap: break-word;'>-1786.3 (0.107)</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>GARCH-X</td><td style='text-align: center; word-wrap: break-word;'>0.508 (0.117)</td><td style='text-align: center; word-wrap: break-word;'>0.221 (0.045)</td><td style='text-align: center; word-wrap: break-word;'>0.195 (0.062)</td><td style='text-align: center; word-wrap: break-word;'>7.807 (1.458)</td><td style='text-align: center; word-wrap: break-word;'></td><td style='text-align: center; word-wrap: break-word;'>-1785.4 (0.321)</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>RECH-X</td><td style='text-align: center; word-wrap: break-word;'>0.272 (0.298)</td><td style='text-align: center; word-wrap: break-word;'>0.199 (0.047)</td><td style='text-align: center; word-wrap: break-word;'>0.216 (0.072)</td><td style='text-align: center; word-wrap: break-word;'>7.663 (1.322)</td><td style='text-align: center; word-wrap: break-word;'>0.294 (0.153)</td><td style='text-align: center; word-wrap: break-word;'>-1781.9 (0.246)</td></tr><tr><td rowspan="3">BVSP</td><td style='text-align: center; word-wrap: break-word;'>GARCH</td><td style='text-align: center; word-wrap: break-word;'>0.108 (0.076)</td><td style='text-align: center; word-wrap: break-word;'>0.051 (0.017)</td><td style='text-align: center; word-wrap: break-word;'>0.861 (0.063)</td><td style='text-align: center; word-wrap: break-word;'>7.056 (1.282)</td><td style='text-align: center; word-wrap: break-word;'></td><td style='text-align: center; word-wrap: break-word;'>-1770.6 (0.064)</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>GARCH-X</td><td style='text-align: center; word-wrap: break-word;'>0.544 (0.181)</td><td style='text-align: center; word-wrap: break-word;'>0.021 (0.025)</td><td style='text-align: center; word-wrap: break-word;'>0.170 (0.019)</td><td style='text-align: center; word-wrap: break-word;'>7.615 (0.149)</td><td style='text-align: center; word-wrap: break-word;'></td><td style='text-align: center; word-wrap: break-word;'>-1766.2 (0.179)</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>RECH-X</td><td style='text-align: center; word-wrap: break-word;'>0.281 (0.292)</td><td style='text-align: center; word-wrap: break-word;'>0.038 (0.018)</td><td style='text-align: center; word-wrap: break-word;'>0.668 (0.164)</td><td style='text-align: center; word-wrap: break-word;'>7.108 (1.211)</td><td style='text-align: center; word-wrap: break-word;'>0.209 (0.088)</td><td style='text-align: center; word-wrap: break-word;'>-1765.3 (0.39)</td></tr></table>

other countries' stock markets will be influenced by the volatility of the US stock market, as indicated by the VIX. The persistence of financial returns volatility further supports the notion that the VIX variable might have a positive effect on the long-term volatility of another nation's stock market, given the high persistence of volatility in financial returns.

We now test the effect of these four exogenous variables on the volatility of the five stock markets mentioned above. As the realized volatilities are not publicly available for those countries, we only compare RECH-X with GARCH-X and GARCH. Table 6 summarizes the in-sample estimation results. Further results on residual analysis can be found in Appendix C. As can be seen, the log marginal likelihood estimates show that the data strongly support the RECH-X model.

Table 7 shows the impact of each individual financial-economic variable. We draw the following conclusions: Firstly, the uncertainty of the US stock market has a positive effect on the stock market volatility of all the five countries, and this effect is significant in the case of CAC40, N225, ASX and BVSP. Among the four exogenous variables, the uncertainty of the US stock market has a positive and the most strong impact, as expected.

Secondly, comparing the coefficients of gold price across the five markets, it can be seen that the volatility in Vietnam, France, Australia, and Brazil markets are quite strongly influenced by the gold price. For France, Australia, and Brazil markets, we observe that gold price has a negative effect, which is consistent with the finding of Garefalakis et al. (2011), while in the other two markets, volatility is positively correlated with the gold price, similar to Akbar et al. (2019) and Singhal et al. (2019). The volatility of the Japanese stock market is less influenced by the gold price.

Thirdly, consistent with the previous studies, the oil price has a mixed effect on the volatility. In France, Japan, Australia and Brazil, the volatility tends to decrease when the oil price increases as studied by Singhal et al. (2019). Particularly, in Australia rising oil prices can cause a significant reduction in volatility in the stock market. Only in Vietnam, oil price and volatility have a positive relationship similar to the results of Mokni and Youssef (2019) and Khan et al. (2021), with a relatively small influence.

Lastly, with the foreign exchange rate, we find that it has a positive influence and has a great impact on the volatility of the Brazilian stock market while the remaining 4 countries have a relatively low influence. In Vietnam and Australia, the correlation is opposite as the conclusion of Adam and Tweneboah (2008), the remaining countries are positively correlated as the conclusion of Maysami et al. (2004).

We now test the combined influence of the financial-economic indicators on volatility forecasting. Table 8 shows that, with the inclusion of these four financial-economic indicators, the RECH-X model delivers a better volatility forecast than the GARCH and GARCH-X model.

<div style="text-align: center;"><div style="text-align: center;">Table 7 Posterior mean and standard deviation of the estimated coefficients corresponding to the covariates in the RECH-X model.</div> </div>




<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>Data</td><td style='text-align: center; word-wrap: break-word;'>OIL</td><td style='text-align: center; word-wrap: break-word;'>GOLD</td><td style='text-align: center; word-wrap: break-word;'>VIX</td><td style='text-align: center; word-wrap: break-word;'>EXR</td></tr><tr><td rowspan="2">VN100</td><td style='text-align: center; word-wrap: break-word;'>0.086</td><td style='text-align: center; word-wrap: break-word;'>0.237</td><td style='text-align: center; word-wrap: break-word;'>0.190</td><td style='text-align: center; word-wrap: break-word;'>$ -0.058 $</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>(0.121)</td><td style='text-align: center; word-wrap: break-word;'>(0.133)</td><td style='text-align: center; word-wrap: break-word;'>(0.115)</td><td style='text-align: center; word-wrap: break-word;'>(0.08)</td></tr><tr><td rowspan="2">CAC40</td><td style='text-align: center; word-wrap: break-word;'>$ -0.115 $</td><td style='text-align: center; word-wrap: break-word;'>$ -0.135 $</td><td style='text-align: center; word-wrap: break-word;'>0.336</td><td style='text-align: center; word-wrap: break-word;'>0.037</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>(0.075)</td><td style='text-align: center; word-wrap: break-word;'>(0.079)</td><td style='text-align: center; word-wrap: break-word;'>(0.137)</td><td style='text-align: center; word-wrap: break-word;'>(0.066)</td></tr><tr><td rowspan="2">N225</td><td style='text-align: center; word-wrap: break-word;'>$ -0.042 $</td><td style='text-align: center; word-wrap: break-word;'>0.004</td><td style='text-align: center; word-wrap: break-word;'>0.569</td><td style='text-align: center; word-wrap: break-word;'>0.014</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>(0.059)</td><td style='text-align: center; word-wrap: break-word;'>(0.046)</td><td style='text-align: center; word-wrap: break-word;'>(0.160)</td><td style='text-align: center; word-wrap: break-word;'>(0.067)</td></tr><tr><td rowspan="2">ASX</td><td style='text-align: center; word-wrap: break-word;'>$ -0.351 $</td><td style='text-align: center; word-wrap: break-word;'>$ -0.374 $</td><td style='text-align: center; word-wrap: break-word;'>0.451</td><td style='text-align: center; word-wrap: break-word;'>$ -0.014 $</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>(0.247)</td><td style='text-align: center; word-wrap: break-word;'>(0.241)</td><td style='text-align: center; word-wrap: break-word;'>(0.273)</td><td style='text-align: center; word-wrap: break-word;'>(0.244)</td></tr><tr><td rowspan="2">BVSP</td><td style='text-align: center; word-wrap: break-word;'>$ -0.096 $</td><td style='text-align: center; word-wrap: break-word;'>$ -0.408 $</td><td style='text-align: center; word-wrap: break-word;'>0.319</td><td style='text-align: center; word-wrap: break-word;'>0.302</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>(0.148)</td><td style='text-align: center; word-wrap: break-word;'>(0.227)</td><td style='text-align: center; word-wrap: break-word;'>(0.162)</td><td style='text-align: center; word-wrap: break-word;'>(0.164)</td></tr></table>

<div style="text-align: center;"><div style="text-align: center;">Table 8</div> </div>


<div style="text-align: center;"><div style="text-align: center;">Predictive performance. The numbers in bold indicate the best values.</div> </div>




<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>Data</td><td style='text-align: center; word-wrap: break-word;'>Model</td><td style='text-align: center; word-wrap: break-word;'>PPS</td><td style='text-align: center; word-wrap: break-word;'>QS</td></tr><tr><td rowspan="3">VN100</td><td style='text-align: center; word-wrap: break-word;'>GARCH</td><td style='text-align: center; word-wrap: break-word;'>1.59</td><td style='text-align: center; word-wrap: break-word;'>0.056</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>GARCH-X</td><td style='text-align: center; word-wrap: break-word;'>1.62</td><td style='text-align: center; word-wrap: break-word;'>0.054</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>RECH-X</td><td style='text-align: center; word-wrap: break-word;'>1.59</td><td style='text-align: center; word-wrap: break-word;'>0.055</td></tr><tr><td rowspan="3">CAC40</td><td style='text-align: center; word-wrap: break-word;'>GARCH</td><td style='text-align: center; word-wrap: break-word;'>1.541</td><td style='text-align: center; word-wrap: break-word;'>0.051</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>GARCH-X</td><td style='text-align: center; word-wrap: break-word;'>1.57</td><td style='text-align: center; word-wrap: break-word;'>0.051</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>RECH-X</td><td style='text-align: center; word-wrap: break-word;'>1.490</td><td style='text-align: center; word-wrap: break-word;'>0.045</td></tr><tr><td rowspan="3">N225</td><td style='text-align: center; word-wrap: break-word;'>GARCH</td><td style='text-align: center; word-wrap: break-word;'>1.622</td><td style='text-align: center; word-wrap: break-word;'>0.047</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>GARCH-X</td><td style='text-align: center; word-wrap: break-word;'>1.67</td><td style='text-align: center; word-wrap: break-word;'>0.054</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>RECH-X</td><td style='text-align: center; word-wrap: break-word;'>1.604</td><td style='text-align: center; word-wrap: break-word;'>0.046</td></tr><tr><td rowspan="3">ASX</td><td style='text-align: center; word-wrap: break-word;'>GARCH</td><td style='text-align: center; word-wrap: break-word;'>2.046</td><td style='text-align: center; word-wrap: break-word;'>0.059</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>GARCH-X</td><td style='text-align: center; word-wrap: break-word;'>2.09</td><td style='text-align: center; word-wrap: break-word;'>0.065</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>RECH-X</td><td style='text-align: center; word-wrap: break-word;'>2.040</td><td style='text-align: center; word-wrap: break-word;'>0.059</td></tr><tr><td rowspan="3">BVSP</td><td style='text-align: center; word-wrap: break-word;'>GARCH</td><td style='text-align: center; word-wrap: break-word;'>1.781</td><td style='text-align: center; word-wrap: break-word;'>0.058</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>GARCH-X</td><td style='text-align: center; word-wrap: break-word;'>1.801</td><td style='text-align: center; word-wrap: break-word;'>0.061</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>RECH-X</td><td style='text-align: center; word-wrap: break-word;'>1.767</td><td style='text-align: center; word-wrap: break-word;'>0.057</td></tr></table>

## 5. Conclusion

This paper proposes a deep learning enhanced volatility model RECH-X that allows for the incorporation of high-frequency data and financial-economic information as exogenous covariates. Bayesian inference and prediction is performed using Sequential Monte Carlo samplers. We find that the RECH-X model using a realized measure as the covariate improves performance compared to GARCH, GARCH-X and RealGARCH. Incorporating financial-economic indicators into the input of RECH-X leads to some interesting findings, and a significant improvement in volatility modeling and forecasting. It is possible to extend the RECH-X model into a mix-frequency context similar to what has been considered by Engle et al. (2013), Asgharian et al. (2013), Conrad and Kleen (2020) and Virbickaite et al. (2023); this research is in progress.

#### CRediT authorship contribution statement

Hien Thi Nguyen: Writing – review & editing, Writing – original draft, Formal analysis. Hoang Nguyen: Writing – review & editing, Writing – original draft, Conceptualization. Minh-Ngoc Tran: Supervision, Conceptualization.

### Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Data availability

The authors do not have permission to share data.

<div style="text-align: center;"><div style="text-align: center;">Table 9 Description of the stock index data.</div> </div>




<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>Symbol</td><td style='text-align: center; word-wrap: break-word;'>Stock name</td><td style='text-align: center; word-wrap: break-word;'>Start date</td><td style='text-align: center; word-wrap: break-word;'>End date</td><td style='text-align: center; word-wrap: break-word;'>Size</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>AEX</td><td style='text-align: center; word-wrap: break-word;'>Amsterdam Exchange Index</td><td style='text-align: center; word-wrap: break-word;'>2012-03-20</td><td style='text-align: center; word-wrap: break-word;'>2020-01-24</td><td style='text-align: center; word-wrap: break-word;'>2001</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>AORD</td><td style='text-align: center; word-wrap: break-word;'>Australia All Ordinaries Index</td><td style='text-align: center; word-wrap: break-word;'>2012-02-28</td><td style='text-align: center; word-wrap: break-word;'>2020-01-24</td><td style='text-align: center; word-wrap: break-word;'>2001</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>BFX</td><td style='text-align: center; word-wrap: break-word;'>Brussels Stock Exchange Index</td><td style='text-align: center; word-wrap: break-word;'>2012-03-20</td><td style='text-align: center; word-wrap: break-word;'>2020-01-24</td><td style='text-align: center; word-wrap: break-word;'>2001</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>BVSP</td><td style='text-align: center; word-wrap: break-word;'>Sao Paulo Stock Exchange</td><td style='text-align: center; word-wrap: break-word;'>2011-12-13</td><td style='text-align: center; word-wrap: break-word;'>2020-01-24</td><td style='text-align: center; word-wrap: break-word;'>2001</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>DJI</td><td style='text-align: center; word-wrap: break-word;'>Dow Jones Industrial Average</td><td style='text-align: center; word-wrap: break-word;'>2012-02-02</td><td style='text-align: center; word-wrap: break-word;'>2020-01-24</td><td style='text-align: center; word-wrap: break-word;'>2001</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>FCHI</td><td style='text-align: center; word-wrap: break-word;'>Euronext Paris CAC 40 Index</td><td style='text-align: center; word-wrap: break-word;'>2012-03-22</td><td style='text-align: center; word-wrap: break-word;'>2020-01-24</td><td style='text-align: center; word-wrap: break-word;'>2001</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>FTSE</td><td style='text-align: center; word-wrap: break-word;'>Financial Times Stock Exchange</td><td style='text-align: center; word-wrap: break-word;'>2012-02-23</td><td style='text-align: center; word-wrap: break-word;'>2020-01-24</td><td style='text-align: center; word-wrap: break-word;'>2001</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>DAX</td><td style='text-align: center; word-wrap: break-word;'>Frankfurt Stock Exchange</td><td style='text-align: center; word-wrap: break-word;'>2012-02-21</td><td style='text-align: center; word-wrap: break-word;'>2020-01-24</td><td style='text-align: center; word-wrap: break-word;'>2001</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>N225</td><td style='text-align: center; word-wrap: break-word;'>Nikkei Stock Average 225</td><td style='text-align: center; word-wrap: break-word;'>2011-11-16</td><td style='text-align: center; word-wrap: break-word;'>2020-01-24</td><td style='text-align: center; word-wrap: break-word;'>2001</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>SPX</td><td style='text-align: center; word-wrap: break-word;'>The Standard and Poor&#x27;s 500</td><td style='text-align: center; word-wrap: break-word;'>2012-02-06</td><td style='text-align: center; word-wrap: break-word;'>2020-01-24</td><td style='text-align: center; word-wrap: break-word;'>2001</td></tr></table>

<div style="text-align: center;"><div style="text-align: center;">Table 10</div> </div>


<div style="text-align: center;"><div style="text-align: center;">Predictive performance of GARCH, RealGARCH and RECH-X across a range of index returns. The numbers in bold indicate the best values.</div> </div>




<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>Data</td><td style='text-align: center; word-wrap: break-word;'>Model</td><td style='text-align: center; word-wrap: break-word;'>PPS</td><td style='text-align: center; word-wrap: break-word;'>QS</td><td style='text-align: center; word-wrap: break-word;'>MSE</td><td style='text-align: center; word-wrap: break-word;'>MAE</td><td style='text-align: center; word-wrap: break-word;'>R2LOG</td><td style='text-align: center; word-wrap: break-word;'>Mar.llh</td></tr><tr><td rowspan="3">DJI</td><td style='text-align: center; word-wrap: break-word;'>GARCH</td><td style='text-align: center; word-wrap: break-word;'>1.249</td><td style='text-align: center; word-wrap: break-word;'>0.034</td><td style='text-align: center; word-wrap: break-word;'>0.132</td><td style='text-align: center; word-wrap: break-word;'>0.282</td><td style='text-align: center; word-wrap: break-word;'>0.877</td><td style='text-align: center; word-wrap: break-word;'>-1498.1</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>RealGARCH</td><td style='text-align: center; word-wrap: break-word;'>1.232</td><td style='text-align: center; word-wrap: break-word;'>0.037</td><td style='text-align: center; word-wrap: break-word;'>0.096</td><td style='text-align: center; word-wrap: break-word;'>0.235</td><td style='text-align: center; word-wrap: break-word;'>0.684</td><td style='text-align: center; word-wrap: break-word;'>-1547.3</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>RECH-X</td><td style='text-align: center; word-wrap: break-word;'>1.191</td><td style='text-align: center; word-wrap: break-word;'>0.032</td><td style='text-align: center; word-wrap: break-word;'>0.095</td><td style='text-align: center; word-wrap: break-word;'>0.226</td><td style='text-align: center; word-wrap: break-word;'>0.567</td><td style='text-align: center; word-wrap: break-word;'>-1446.5</td></tr><tr><td rowspan="3">AORD</td><td style='text-align: center; word-wrap: break-word;'>GARCH</td><td style='text-align: center; word-wrap: break-word;'>1.022</td><td style='text-align: center; word-wrap: break-word;'>0.029</td><td style='text-align: center; word-wrap: break-word;'>0.093</td><td style='text-align: center; word-wrap: break-word;'>0.259</td><td style='text-align: center; word-wrap: break-word;'>1.098</td><td style='text-align: center; word-wrap: break-word;'>-1639.5</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>RealGARCH</td><td style='text-align: center; word-wrap: break-word;'>1.033</td><td style='text-align: center; word-wrap: break-word;'>0.028</td><td style='text-align: center; word-wrap: break-word;'>0.097</td><td style='text-align: center; word-wrap: break-word;'>0.274</td><td style='text-align: center; word-wrap: break-word;'>1.196</td><td style='text-align: center; word-wrap: break-word;'>-1636.3</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>RECH-X</td><td style='text-align: center; word-wrap: break-word;'>1.010</td><td style='text-align: center; word-wrap: break-word;'>0.027</td><td style='text-align: center; word-wrap: break-word;'>0.086</td><td style='text-align: center; word-wrap: break-word;'>0.240</td><td style='text-align: center; word-wrap: break-word;'>0.963</td><td style='text-align: center; word-wrap: break-word;'>-1618.8</td></tr><tr><td rowspan="3">FTSE</td><td style='text-align: center; word-wrap: break-word;'>GARCH</td><td style='text-align: center; word-wrap: break-word;'>1.195</td><td style='text-align: center; word-wrap: break-word;'>0.029</td><td style='text-align: center; word-wrap: break-word;'>0.094</td><td style='text-align: center; word-wrap: break-word;'>0.232</td><td style='text-align: center; word-wrap: break-word;'>0.656</td><td style='text-align: center; word-wrap: break-word;'>-1743.7</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>RealGARCH</td><td style='text-align: center; word-wrap: break-word;'>1.181</td><td style='text-align: center; word-wrap: break-word;'>0.028</td><td style='text-align: center; word-wrap: break-word;'>0.076</td><td style='text-align: center; word-wrap: break-word;'>0.205</td><td style='text-align: center; word-wrap: break-word;'>0.567</td><td style='text-align: center; word-wrap: break-word;'>-1738.7</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>RECH-X</td><td style='text-align: center; word-wrap: break-word;'>1.166</td><td style='text-align: center; word-wrap: break-word;'>0.027</td><td style='text-align: center; word-wrap: break-word;'>0.077</td><td style='text-align: center; word-wrap: break-word;'>0.201</td><td style='text-align: center; word-wrap: break-word;'>0.500</td><td style='text-align: center; word-wrap: break-word;'>-1707.0</td></tr><tr><td rowspan="3">BVSP</td><td style='text-align: center; word-wrap: break-word;'>GARCH</td><td style='text-align: center; word-wrap: break-word;'>1.655</td><td style='text-align: center; word-wrap: break-word;'>0.040</td><td style='text-align: center; word-wrap: break-word;'>0.302</td><td style='text-align: center; word-wrap: break-word;'>0.510</td><td style='text-align: center; word-wrap: break-word;'>1.365</td><td style='text-align: center; word-wrap: break-word;'>-2636.1</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>RealGARCH</td><td style='text-align: center; word-wrap: break-word;'>1.642</td><td style='text-align: center; word-wrap: break-word;'>0.039</td><td style='text-align: center; word-wrap: break-word;'>0.286</td><td style='text-align: center; word-wrap: break-word;'>0.491</td><td style='text-align: center; word-wrap: break-word;'>1.242</td><td style='text-align: center; word-wrap: break-word;'>-2613.6</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>RECH-X</td><td style='text-align: center; word-wrap: break-word;'>1.639</td><td style='text-align: center; word-wrap: break-word;'>0.039</td><td style='text-align: center; word-wrap: break-word;'>0.278</td><td style='text-align: center; word-wrap: break-word;'>0.484</td><td style='text-align: center; word-wrap: break-word;'>1.207</td><td style='text-align: center; word-wrap: break-word;'>-2620.0</td></tr><tr><td rowspan="3">BFX</td><td style='text-align: center; word-wrap: break-word;'>GARCH</td><td style='text-align: center; word-wrap: break-word;'>1.256</td><td style='text-align: center; word-wrap: break-word;'>0.030</td><td style='text-align: center; word-wrap: break-word;'>0.091</td><td style='text-align: center; word-wrap: break-word;'>0.254</td><td style='text-align: center; word-wrap: break-word;'>0.663</td><td style='text-align: center; word-wrap: break-word;'>-1919.1</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>RealGARCH</td><td style='text-align: center; word-wrap: break-word;'>1.278</td><td style='text-align: center; word-wrap: break-word;'>0.030</td><td style='text-align: center; word-wrap: break-word;'>0.098</td><td style='text-align: center; word-wrap: break-word;'>0.280</td><td style='text-align: center; word-wrap: break-word;'>0.805</td><td style='text-align: center; word-wrap: break-word;'>-1913.6</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>RECH-X</td><td style='text-align: center; word-wrap: break-word;'>1.222</td><td style='text-align: center; word-wrap: break-word;'>0.028</td><td style='text-align: center; word-wrap: break-word;'>0.089</td><td style='text-align: center; word-wrap: break-word;'>0.239</td><td style='text-align: center; word-wrap: break-word;'>0.608</td><td style='text-align: center; word-wrap: break-word;'>-1878.4</td></tr><tr><td rowspan="3">AEX</td><td style='text-align: center; word-wrap: break-word;'>GARCH</td><td style='text-align: center; word-wrap: break-word;'>1.212</td><td style='text-align: center; word-wrap: break-word;'>0.034</td><td style='text-align: center; word-wrap: break-word;'>0.108</td><td style='text-align: center; word-wrap: break-word;'>0.279</td><td style='text-align: center; word-wrap: break-word;'>0.898</td><td style='text-align: center; word-wrap: break-word;'>-1944.4</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>RealGARCH</td><td style='text-align: center; word-wrap: break-word;'>1.243</td><td style='text-align: center; word-wrap: break-word;'>0.032</td><td style='text-align: center; word-wrap: break-word;'>0.124</td><td style='text-align: center; word-wrap: break-word;'>0.322</td><td style='text-align: center; word-wrap: break-word;'>1.134</td><td style='text-align: center; word-wrap: break-word;'>-1942.5</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>RECH-X</td><td style='text-align: center; word-wrap: break-word;'>1.142</td><td style='text-align: center; word-wrap: break-word;'>0.027</td><td style='text-align: center; word-wrap: break-word;'>0.090</td><td style='text-align: center; word-wrap: break-word;'>0.239</td><td style='text-align: center; word-wrap: break-word;'>0.656</td><td style='text-align: center; word-wrap: break-word;'>-1890.6</td></tr><tr><td rowspan="3">FCHI</td><td style='text-align: center; word-wrap: break-word;'>GARCH</td><td style='text-align: center; word-wrap: break-word;'>1.276</td><td style='text-align: center; word-wrap: break-word;'>0.034</td><td style='text-align: center; word-wrap: break-word;'>0.133</td><td style='text-align: center; word-wrap: break-word;'>0.312</td><td style='text-align: center; word-wrap: break-word;'>0.977</td><td style='text-align: center; word-wrap: break-word;'>-2157.0</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>RealGARCH</td><td style='text-align: center; word-wrap: break-word;'>1.272</td><td style='text-align: center; word-wrap: break-word;'>0.033</td><td style='text-align: center; word-wrap: break-word;'>0.117</td><td style='text-align: center; word-wrap: break-word;'>0.306</td><td style='text-align: center; word-wrap: break-word;'>0.982</td><td style='text-align: center; word-wrap: break-word;'>-2131.4</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>RECH-X</td><td style='text-align: center; word-wrap: break-word;'>1.204</td><td style='text-align: center; word-wrap: break-word;'>0.029</td><td style='text-align: center; word-wrap: break-word;'>0.105</td><td style='text-align: center; word-wrap: break-word;'>0.259</td><td style='text-align: center; word-wrap: break-word;'>0.674</td><td style='text-align: center; word-wrap: break-word;'>-2100.1</td></tr><tr><td rowspan="3">DAX</td><td style='text-align: center; word-wrap: break-word;'>GARCH</td><td style='text-align: center; word-wrap: break-word;'>1.387</td><td style='text-align: center; word-wrap: break-word;'>0.033</td><td style='text-align: center; word-wrap: break-word;'>0.148</td><td style='text-align: center; word-wrap: break-word;'>0.335</td><td style='text-align: center; word-wrap: break-word;'>0.939</td><td style='text-align: center; word-wrap: break-word;'>-2178.4</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>RealGARCH</td><td style='text-align: center; word-wrap: break-word;'>1.359</td><td style='text-align: center; word-wrap: break-word;'>0.032</td><td style='text-align: center; word-wrap: break-word;'>0.104</td><td style='text-align: center; word-wrap: break-word;'>0.287</td><td style='text-align: center; word-wrap: break-word;'>0.772</td><td style='text-align: center; word-wrap: break-word;'>-2146.3</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>RECH-X</td><td style='text-align: center; word-wrap: break-word;'>1.334</td><td style='text-align: center; word-wrap: break-word;'>0.031</td><td style='text-align: center; word-wrap: break-word;'>0.113</td><td style='text-align: center; word-wrap: break-word;'>0.281</td><td style='text-align: center; word-wrap: break-word;'>0.674</td><td style='text-align: center; word-wrap: break-word;'>-2131.9</td></tr></table>

### Appendix A. Forecast evaluation

In order to assess the predictive performance, we use five predictive scores for both point and density forecasts. The first is the partial predictive score (PPS),

 $$ \mathrm{P P S}:=-\frac{1}{T_{\mathrm{t e s t}}}\sum_{t=T+1}^{T+T_{\mathrm{t e s t}}}\log p(y_{t}|y_{1:t-1}), $$ 

where  $ T_{test} $ is the number of observations in the test data. The second is quantile score defined as (Taylor, 2019)

 $$ \mathrm{Q S}:=\frac{1}{T_{\mathrm{t e s t}}}\sum_{t=T+1}^{T+T_{\mathrm{t e s t}}}(\alpha-I_{y_{t}\leq q_{t,\alpha}})(y_{t}-q_{t,\alpha}), $$ 

where  $ q_{t,\alpha} $ is the  $ \alpha $-VaR forecast of  $ y_t $, conditional on  $ y_{1:t-1} $. The smaller the quantile score, the better the Value-at-Risk forecast. The next three predictive scores compare the model-based volatility forecast  $ \hat{v}_t^2 $ with the realized volatility in the test data (Hansen and Lunde, 2005):

 $$ \mathrm{MSE}=\frac{1}{T_{\mathrm{test}}}\sum_{t=T+1}^{T+T_{\mathrm{test}}}(\mathrm{RV}_{t}^{1/2}-\widehat{v}_{t})^{2}, $$ 

 $$  MAE=\frac{1}{T_{test}}\sum_{t=T+1}^{T+T_{test}}|(\mathrm{RV}_{t}^{1/2}-\widehat{v}_{t})|, $$ 

 $$ \mathrm{R}^{2}\mathrm{L O G}=\frac{1}{T_{\mathrm{t e s t}}}\sum_{t=T+1}^{T+T_{\mathrm{t e s t}}}\left[\log(\mathrm{R V}_{t}\widehat{v}_{t}^{-2})\right]^{2}, $$ 

where  $ \widehat{v}_t^2 = v\widehat{\sigma}_t^2 / (v - 2) $ with  $ \widehat{\sigma}_t^2 $ a forecast of  $ \sigma_t^2 $ using the volatility model with Student's  $ t $ innovations.

## Appendix B. Predictive performance of the RECH-X model with realized volatility

This section assesses the predictive performance of the three models on a wide range of index returns. We opt to not present the in-sample estimation results, except the log marginal likelihood estimates listed in the last column of Table 10. The RECH-X model has the highest log marginal likelihood estimates in all cases except for the BVSP data. The results in Table 10 show that the RECH-X model consistently has an impressive predictive performance across all the datasets. The RealGARCH model is, in general, better than GARCH.

## Appendix C. Predictive performance of the RECH-X model with financial-economic variables

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//792cd0e9-7a76-44f4-bbae-fd6f4770b53e/markdown_0/imgs/img_in_chart_box_109_443_979_845.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-05-15T14%3A48%3A09Z%2F-1%2F%2Fa2ce04bedb22f547c57d4c173357b52e486572b87bde593c0aa7c4c1bb4d10f0" alt="Image" width="79%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Fig. 3. VN100: plots of return indices in the testing data together with 95% one-step-ahead forecast intervals from GARCH (dashed green) and RECH-X (solid black).</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//792cd0e9-7a76-44f4-bbae-fd6f4770b53e/markdown_0/imgs/img_in_chart_box_138_922_525_1133.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-05-15T14%3A48%3A09Z%2F-1%2F%2F031f0b6bc81d266a62f3d7c64c841242750cea473e78c5b7d3a5a93a7fa55ce0" alt="Image" width="35%" /></div>


<div style="text-align: center;"><div style="text-align: center;">QQ Plot of Sample Data versus Standard Normal (GARCH - residuals)</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//792cd0e9-7a76-44f4-bbae-fd6f4770b53e/markdown_0/imgs/img_in_chart_box_583_940_980_1134.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-05-15T14%3A48%3A10Z%2F-1%2F%2F57a6c7f05c47cabf728f223a8e860f1a9289ce1d641e5f5fbf891b02c931fe31" alt="Image" width="36%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//792cd0e9-7a76-44f4-bbae-fd6f4770b53e/markdown_0/imgs/img_in_chart_box_142_1146_521_1339.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-05-15T14%3A48%3A10Z%2F-1%2F%2F6ebaeb53c54ef7897e9be4e4604a2aa9c63cf70a7aba528ef30abdc4a29a72fd" alt="Image" width="34%" /></div>


<div style="text-align: center;"><div style="text-align: center;">QQ Plot of Sample Data versus Standard Normal (RECH-X - residuals)</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//792cd0e9-7a76-44f4-bbae-fd6f4770b53e/markdown_0/imgs/img_in_chart_box_584_1154_976_1348.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-05-15T14%3A48%3A10Z%2F-1%2F%2F07630ecca37b3b5ec9fa1947d3c2db174c051dc0dd4f4a76be1039681ca4c23d" alt="Image" width="35%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Fig. 4. VN100: plots of sample data versus standard normal.</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//792cd0e9-7a76-44f4-bbae-fd6f4770b53e/markdown_1/imgs/img_in_chart_box_107_149_977_601.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-05-15T14%3A48%3A11Z%2F-1%2F%2F04343998362aac7212f21d0f4e220b9a2b06fcc0ccd3e1760511f2f136ea6e50" alt="Image" width="79%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Fig. 5. CAC40: plots of return indices in the testing data together with 95% one-step-ahead forecast intervals from GARCH (dashed green) and RECH-X (solid black).</div> </div>


<div style="text-align: center;"><div style="text-align: center;">CAC40</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//792cd0e9-7a76-44f4-bbae-fd6f4770b53e/markdown_1/imgs/img_in_chart_box_138_727_529_958.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-05-15T14%3A48%3A11Z%2F-1%2F%2Ff0f8716ee389490c468ee259ab4ba88915fbf48b931e0cfb611d9961a230059f" alt="Image" width="35%" /></div>


<div style="text-align: center;"><div style="text-align: center;">QQ Plot of Sample Data versus Standard Normal (GARCH - residuals)</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//792cd0e9-7a76-44f4-bbae-fd6f4770b53e/markdown_1/imgs/img_in_chart_box_587_746_984_960.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-05-15T14%3A48%3A11Z%2F-1%2F%2F855a8cd0ac91213d6be60113230e100f0f6548a0488d6e9ecf44e2e6bf5d0ac6" alt="Image" width="36%" /></div>


<div style="text-align: center;"><div style="text-align: center;">QQ Plot of Sample Data versus Standard Normal (RECH-X - residuals)</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//792cd0e9-7a76-44f4-bbae-fd6f4770b53e/markdown_1/imgs/img_in_chart_box_141_970_527_1190.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-05-15T14%3A48%3A12Z%2F-1%2F%2Fcff6a4d0ef3197de7bc5274070ebe4f5b433bbee2267b44c77433432e0f6244c" alt="Image" width="35%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//792cd0e9-7a76-44f4-bbae-fd6f4770b53e/markdown_1/imgs/img_in_chart_box_588_984_983_1195.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-05-15T14%3A48%3A12Z%2F-1%2F%2F096d177f42695cad63c0c8d67d2ee6be3e176be5829ee3d9dd5a95ea32f44b45" alt="Image" width="36%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Fig. 6. CAC40: plots of sample data versus standard normal.</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//792cd0e9-7a76-44f4-bbae-fd6f4770b53e/markdown_2/imgs/img_in_chart_box_108_186_977_625.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-05-15T14%3A48%3A13Z%2F-1%2F%2Fd2a7160189d0dc0031b8b10481c2b125fc0007c9160db67a14fefc2487be28b7" alt="Image" width="79%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Fig. 7. N225: plots of return indices in the testing data together with 95% one-step-ahead forecast intervals from GARCH (dashed green) and RECH-X (solid black).</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//792cd0e9-7a76-44f4-bbae-fd6f4770b53e/markdown_2/imgs/img_in_chart_box_101_802_531_1039.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-05-15T14%3A48%3A13Z%2F-1%2F%2Fcd847cc03f7dea305c29331e17f4c1022481fcf480a36de2b3b4e2005ca2d6d4" alt="Image" width="39%" /></div>


<div style="text-align: center;"><div style="text-align: center;">QQ Plot of Sample Data versus Standard Normal (GARCH - residuals)</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//792cd0e9-7a76-44f4-bbae-fd6f4770b53e/markdown_2/imgs/img_in_chart_box_587_822_985_1038.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-05-15T14%3A48%3A13Z%2F-1%2F%2Fe413484891af40e0c9c152f3095bcacd3697275e18663eef3b98be038f673478" alt="Image" width="36%" /></div>


<div style="text-align: center;"><div style="text-align: center;">QQ Plot of Sample Data versus Standard Normal (RECH-X - residuals)</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//792cd0e9-7a76-44f4-bbae-fd6f4770b53e/markdown_2/imgs/img_in_chart_box_142_1054_527_1270.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-05-15T14%3A48%3A13Z%2F-1%2F%2F7abf9bbc27d517c0138917b188ac15c1d8dd0613dc04e73264eeeb35e879f189" alt="Image" width="35%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//792cd0e9-7a76-44f4-bbae-fd6f4770b53e/markdown_2/imgs/img_in_chart_box_587_1062_984_1272.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-05-15T14%3A48%3A14Z%2F-1%2F%2F21ed4507dd47763c890e60b906011807dc1444f72bb254a2352e5b6f487d73e9" alt="Image" width="36%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Fig. 8. N225: plots of sample data versus standard normal.</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//792cd0e9-7a76-44f4-bbae-fd6f4770b53e/markdown_3/imgs/img_in_chart_box_107_112_978_565.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-05-15T14%3A48%3A15Z%2F-1%2F%2F817d0d317fb9882168e148f99c90c3f039b8c9fef40b376a52ada0325f505ad9" alt="Image" width="79%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Fig. 9. ASX: plots of return indices in the testing data together with 95% one-step-ahead forecast intervals from GARCH (dashed green) and RECH-X (solid black).</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//792cd0e9-7a76-44f4-bbae-fd6f4770b53e/markdown_3/imgs/img_in_chart_box_136_635_529_867.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-05-15T14%3A48%3A15Z%2F-1%2F%2F592f7f029315881ad1a93473d304138ee12dcca2e218695b03d984fd85bf7052" alt="Image" width="36%" /></div>


<div style="text-align: center;"><div style="text-align: center;">QQ Plot of Sample Data versus Standard Normal (GARCH - residuals)</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//792cd0e9-7a76-44f4-bbae-fd6f4770b53e/markdown_3/imgs/img_in_chart_box_588_648_987_867.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-05-15T14%3A48%3A15Z%2F-1%2F%2Fc4cb444aa149bed0da9a49475e52823e372d2c16b3926b267449e42599f9c09f" alt="Image" width="36%" /></div>


<div style="text-align: center;"><div style="text-align: center;">QQ Plot of Sample Data versus Standard Normal (RECH-X - residuals)</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//792cd0e9-7a76-44f4-bbae-fd6f4770b53e/markdown_3/imgs/img_in_chart_box_143_887_528_1099.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-05-15T14%3A48%3A16Z%2F-1%2F%2Fd49665bdad74812c085db08e093671dfdc80ae0966b512ecf908f4a46bb3e3cd" alt="Image" width="35%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-15//792cd0e9-7a76-44f4-bbae-fd6f4770b53e/markdown_3/imgs/img_in_chart_box_589_888_985_1099.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-05-15T14%3A48%3A16Z%2F-1%2F%2F34f4c884400227ab56a47834fad686c8c7d819c06e29860be88a5e8c26c3d2ca" alt="Image" width="36%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Fig. 10. ASX: plots of sample data versus standard normal.</div> </div>


## References

Abugri, B.A., 2008. Empirical relationship between macroeconomic volatility and stock returns: Evidence from Latin American markets. Int. Rev. Financ. Anal. 17 (2), 396–410, URL https://www.sciencedirect.com/science/article/pii/S1057521906000731.

Adam, A.M., Tweneboah, G., 2008. Macroeconomic factors and stock market movement: Evidence from ghana. Available at SSRN 1289842.

Akbar, M., Iqbal, F., Noor, F., 2019. Bayesian analysis of dynamic linkages among gold price, stock prices, exchange rate and interest rate in Pakistan. Resour. Policy 62, 154–164.

Alam, M.M., Uddin, G., 2009. Relationship between interest rate and stock price: empirical evidence from developed and developing countries. Int. J. Bus. Manag. (ISSN: 1833-3850) 4 (3), 43–51.

Andersen, T.G., Bollerslev, T., 1998. Answering the skeptics: Yes, standard volatility models do provide accurate forecasts. Internat. Econom. Rev. 39 (4), 885–905.

Andersen, T.G., Bollerslev, T., Diebold, F.X., Ebens, H., 2001. The distribution of realized stock return volatility. J. Financ. Econ. 61 (1), 43–76.

Asgharian, H., Hou, A.J., Javed, F., 2013. The importance of the macroeconomic variables in forecasting stock return variance: A GARCH-MIDAS approach. J. Forecast. 32 (7), 600–612.

Aydemir, O., Demirhan, E., 2009. The relationship between stock prices and exchange rates: Evidence from Turkey. Int. Res. J. Finance Econ. 23 (2), 207–215.

Barndorff-Nielsen, O.E., Shephard, N., 2002. Estimating quadratic variation using realized variance. J. Appl. Econ. 17 (5), 457–477.

Bekhet, H.A., Mugableh, M.I., 2012. Investigating equilibrium relationship between macroeconomic variables and Malaysian stock market index through bounds tests approach. Int. J. Econ. Finance 4 (10), 69–81.

Black, F., 1976. Studies of stock market volatility changes. In: 1976 Proceedings of the American Statistical Association Business and Economic Statistics Section. American Statistical Association.

Bollerslev, T., 1986. Generalized autoregressive conditional heteroskedasticity. J. Econometrics 31 (3), 307–327.

Bollerslev, T., Melvin, M., 1994. Bid—ask spreads and volatility in the foreign exchange market: An empirical analysis. J. Int. Econ. 36 (3–4), 355–372.

Brenner, R.J., Harjes, R.H., Kroner, K.F., 1996. Another look at models of the short-term interest rate. J. Financ. Quant. Anal. 31 (1), 85–107.

Carnero, M.A., Peña, D., Ruiz, E., 2004. Persistence and kurtosis in GARCH and stochastic volatility models. J. Financ. Econom. 2 (2), 319–342.

Conrad, C., Kleen, O., 2020. Two are better than one: Volatility forecasting using multiplicative component GARCH-MIDAS models. J. Appl. Econometrics 35(1), 19–45.

Duan, J.-C., Fulop, A., 2015. Density-tempered marginalized sequential Monte Carlo samplers. J. Bus. Econom. Statist. 33 (2), 192–202, arXiv:https://doi.org/10.1080/07350015.2014.940081.

Engle, R.F., 1982. A general approach to Lagrange multiplier model diagnostics. J. Econometrics 20 (1), 83–104.

Engle, R., 2002. New frontiers for ARCH models. J. Appl. Econometrics 17 (5), 425–446.

Engle, R.F., Gallo, G.M., 2006. A multiple indicators model for volatility using intra-daily data. J. Econometrics 131 (1–2), 3–27.

Engle, R.F., Ghysels, E., Sohn, B., 2013. Stock market volatility and macroeconomic fundamentals. Rev. Econ. Stat. 95 (3), 776–797.

Engle, R.F., Patton, A.J., 2001. What good is a volatility model? Quant. Finance 1 (2), 237.

Engle, R.F., Rangel, J.G., 2008. The Spline-GARCH model for low-frequency volatility and its global macroeconomic causes. Rev. Financ. Stud. 21 (3), 1187–1222.

Francq, C., Thieu, L.Q., 2019. QML inference for volatility models with covariates. Econometric Theory 35 (1), 37–72.

Garefalakis, A., Dimitras, A., Koemtzopoulos, D., Spinthiroopoulos, K., 2011. Determinant factors of Hong Kong stock market. Int. Res. J. Finance Econ.

Gerlach, R., Wang, C., 2016. Forecasting risk via realized GARCH, incorporating the realized range. Quant. Finance 16 (4), 501–511, arXiv:https://doi.org/10.1080/14697688.2015.1079641.

Glosten, L.R., Jagannathan, R., Runkle, D.E., 1993a. On the relation between the expected value and the volatility of the nominal excess return on stocks. J. Finance 48 (5), 1779–1801.

Glosten, L.R., Jagannathan, R., Runkle, D.E., 1993b. On the relation between the expected value and the volatility of the nominal excess return on stocks. J. Finance 48 (5), 1779–1801, arXiv:https://onlinelibrary.wiley.com/doi/pdf/10.1111/j.1540-6261.1993.tb05128.x. URL https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1993.tb05128.x.

Goodfellow, I., Bengio, Y., Courville, A., 2016. Deep Learning. MIT Press.

Gray, S.F., 1996. Modeling the conditional distribution of interest rates as a regime-switching process. J. Financ. Econ. 42 (1), 27–62.

Gunawan, D., Kohn, R., Tran, M.N., 2022. Flexible and robust particle tempering for state space models. Econ. Stat.

Hagiwara, M., Herce, M.A., 1999. Endogenous exchange rate volatility, trading volume and interest rate differentials in a model of portfolio selection. Rev. Int. Econ. 7 (2), 202–218.

Hajizadeh, E., Seifi, A., Zarandi, M.F., Turksen, I., 2012. A hybrid modeling approach for forecasting the volatility of S&P 500 index return. Expert Syst. Appl. 39 (1), 431–436.

Han, H., 2015. Asymptotic properties of GARCH-X processes. J. Financ. Econom. 13 (1), 188–221.

Han, H., Park, J.Y., 2008. Time series properties of ARCH processes with persistent covariates. J. Econometrics 146 (2), 275–292.

Hansen, P.R., Huang, Z., Shek, H.H., 2012. Realized GARCH: a joint model for returns and realized measures of volatility. J. Appl. Econometrics 27 (6), 877–906,

arXiv:https://onlinelibrary.wiley.com/doi/pdf/10.1002/jae.1234. URL https://onlinelibrary.wiley.com/doi/abs/10.1002/jae.1234.

Hansen, P.R., Lunde, A., 2005. A forecast comparison of volatility models: does anything beat a GARCH (1, 1)? J. Appl. Econometrics 20 (7), 873–889.

Hansen, P.R., Lunde, A., Voev, V., 2014. Realized beta GARCH: A multivariate GARCH model with realized measures of volatility. J. Appl. Econometrics 29 (5), 774–799.

Hochreiter, S., Schmidhuber, J., 1997. Long short-term memory. Neural Comput. 9 (8), 1735–1780.

Hodrick, R.J., 1989. Risk, uncertainty, and exchange rates. J. Monet. Econ. 23 (3), 433–459.

Humpe, A., Macmillan, P., 2007. Can macroeconomic variables explain long term stock market movements? A comparison of the US and Japan CDMA working. J. Finance 30, 209–245.

Kandir, S.Y., 2008. Macroeconomic variables, firm characteristics and stock returns: Evidence from Turkey. Int. Res. J. Finance Econ. 16 (1), 35–45.

Kasman, S., 2003. The relationship between exchange rates and stock prices: A causality analysis.

Khan, M.H., 2014. An empirical investigation on behavioral determinants of perceived investment performance: Evidence from Karachi stock exchange. Res. J. Finance Account. 5 (21), 129–137.

Khan, M.I., Teng, J.-Z., Khan, M.K., Jadoon, A.U., Khan, M.F., 2021. The impact of oil prices on stock market development in Pakistan: Evidence with a novel dynamic simulated ARDL approach. Resour. Policy 70, 101899.

Kim, H.Y., Won, C.H., 2018. Forecasting the volatility of stock price index: A hybrid model integrating LSTM with multiple GARCH-type models. Expert Syst. Appl. 103, 25–37.

Koopman, S.J., Jungbacker, B., Hol, E., 2005. Forecasting daily variability of the S&P 500 stock index using historical, realised and implied volatility measurements. J. Empir. Financ. 12 (3), 445–475.

Kristjanpoller, W., Fadic, A., Minutolo, M.C., 2014. Volatility forecast using hybrid neural network models. Expert Syst. Appl. 41 (5), 2437–2442.

Le, T.M.H., Zhihong, J., Zhu, Z., 2019. Impact of macroeconomic variables on stock price index: Evidence from Vietnam stock market. Res. J. Finance Account. 10 (12), 28–29.

Lipton, Z.C., Berkowitz, J., Elkan, C., 2015. A critical review of recurrent neural networks for sequence learning. arXiv preprint arXiv:1506.00019.

Liu, C., Wang, C., Tran, M.-N., Kohn, R., 2023. Realized recurrent conditional heteroskedasticity model for volatility modelling. arXiv preprint arXiv:2302.08002.

Luo, J., Klein, T., Ji, Q., Hou, C., 2022. Forecasting realized volatility of agricultural commodity futures with infinite hidden Markov HAR models. Int. J. Forecast. 38 (1), 51–73.

Luo, J., Klein, T., Walther, T., Ji, Q., 2021. Forecasting realized volatility of crude oil futures prices based on machine learning. J. Forecast.

Mandelbrot, B., 1967. The variation of some other speculative prices. J. Bus. 40 (4), 393–413.

Maysami, R.C., Howe, L.C., Hamzah, M.A., 2004. Relationship between macroeconomic variables and stock market indices: Cointegration evidence from stock exchange of Singapore's All-S sector indices. J. Pengurusan 24 (1), 47–77.

Mokni, K., Youssef, M., 2019. Measuring persistence of dependence between crude oil prices and GCC stock markets: A copula approach. Q. Rev. Econ. Finance 72, 14–33.

Nguyen, T.-N., Tran, M.-N., Kohn, R., 2022. Recurrent conditional heteroskedasticity. J. Appl. Econometrics 37 (5), 1031–1054.

Roh, T.H., 2007. Forecasting the volatility of stock price index. Expert Syst. Appl. 33 (4), 916–922.

Shephard, N., Sheppard, K., 2010. Realising the future: forecasting with high-frequency-based volatility (HEAVY) models. J. Appl. Econometrics 25 (2), 197–231.

Singhal, S., Choudhary, S., Biswal, P.C., 2019. Return and volatility linkages among international crude oil price, gold price, exchange rate and stock markets: Evidence from Mexico. Resour. Policy 60, 255–261.

Smith, G., 2001. The price of gold and stock price indices for the United States. World Gold Council. 8 (1), 1–16.

Smyth, R., Narayan, P.K., 2018. What do we know about oil prices and stock returns? Int. Rev. Financ. Anal. 57, 148–156.

Taylor, S.J., 1982. Financial returns modelled by the product of two stochastic processes—a study of the daily sugar prices 1961-75. Time Ser. Anal.: Theory Pract. 1, 203–226.

Taylor, J.W., 2019. Forecasting value at risk and expected shortfall using a semiparametric approach based on the asymmetric Laplace distribution. J. Bus. Econom. Statist. 37 (1), 121–133.

Tursoy, T., Gunsel, N., Rjoub, H., 2008. Macroeconomic factors, the APT and the Istanbul stock market. Int. Res. J. Finance Econ. 22 (9).

Virbickaite, A., Nguyen, H., Tran, M.-N., 2023. Bayesian predictive distributions of oil returns using mixed data sampling volatility models. Available at SSRN 4462554.

