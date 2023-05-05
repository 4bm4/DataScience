import pandas as pd
import numpy as np
import random
import inspect
import warnings
import seaborn as sns
from pygam import LinearGAM, s, f, l
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.optimize as stats2
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.genmod.generalized_linear_model import GLMResults
from statsmodels.formula.api import ols,glm
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.metrics import mean_squared_error,r2_score, confusion_matrix, precision_recall_fscore_support,roc_curve, accuracy_score, roc_auc_score
from dmba import stepwise_selection,AIC_score,classificationSummary,textDecisionTree

warnings.filterwarnings('ignore')




class DF_exploracion(pd.DataFrame):

    def __init__(self, *args, **kw):
        super(DF_exploracion, self).__init__(*args, **kw)
        self.cuanti=pd.DataFrame
        self.cuanti_antes_de_outliers_y_inputs=pd.DataFrame
        self.cuali=pd.DataFrame
        self.dico=pd.DataFrame
        self.cate=pd.DataFrame
        self.eliminado=pd.DataFrame
        self.dummy=pd.DataFrame
        self.df=pd.DataFrame
        self.df_inputado=pd.DataFrame
        self.df_limpio=pd.DataFrame
        self.predicotres=pd.DataFrame
        self.outcome=pd.DataFrame
        # self.outcome_col=self.outcome.columns
        self.normal_cuatis=[]
        self.normal_grupos_dico=[]
        self.normal_grupos_cate=[]
        self.discreta=[]
        self.stingg=[]
        self.outliers_hecho=True
        self.porcentaje_nulos_permitido=0.3

    def variables(self):

        dico=[]
        cuantis=[]
        categori=[]
        eliminar=[]
        

        for i in self.columns: 

            try:
                datos=self[i].dropna().to_numpy()
                discreta=True
                for j in datos:
                    if (j%1 !=0):
                        discreta=False
                        break
                    else:
                        continue
                if (discreta):
                    self.discreta.append(i)
            except:
                self.stingg.append(i)

            nulos= (self[i].isnull().sum())/len(self[i])
            
            if ((len(self[i].dropna().unique())==2) and (nulos<=self.porcentaje_nulos_permitido)):
                tipo_de_var=f"{len(self[i].dropna().unique())} tipos, posiblemente: DICOTOMICA"
                dico.append(i)

            elif ((len(self[i].dropna().unique())>10) and  (nulos<=self.porcentaje_nulos_permitido)):
                tipo_de_var=f"{len(self[i].dropna().unique())} tipos, posiblemente: CUANTITATIVA"
                cuantis.append(i)

            elif ( (len(self[i].dropna().unique())<2) or (nulos>self.porcentaje_nulos_permitido)):
                tipo_de_var=f"SOLO {len(self[i].dropna().unique())} TIPOS, NO VALE LA COLUMNA"
                eliminar.append(i)
            else:
                tipo_de_var=f"{len(self[i].dropna().unique())} tipos, posiblemente: CATEGORICA/CUANTI"
                categori.append(i)

            print (f"|  {i} \n|   - Tipo de dato: {self[i].dtype} \n|   - Valores repetidos: {tipo_de_var} \n|   - Nulos: {nulos} \n| ")

        print (f"|----------------------------------------------------------------------------------------------------\n|  TODAS: {self.columns} \n|  DICOTOMICAS: {dico} \n|  CATEGORICAS: {categori} \n|  CUANTITATIVAS: {cuantis} \n|  ELIMINAR: {eliminar}")
        print("|----------------------------------------------------------------------------------------------------")

        


        self.DF_cuantis(cuantis)
        self.DF_cualis(categori+dico)
        self.DF_dicotomica(dico)
        self.DF_categorica(categori)
        self.DF_elimiminado(eliminar)
        self.df=self
        
    def todas_col(self):
        return self.df
    
    def DF_cuantis(self,lista):
        self.cuanti=self[lista]

    def DF_elimiminado(self,lista):
        self.eliminado=self[lista]
        
    def DF_cualis(self,lista):
        self.cuali=self[lista]
        
    def DF_dicotomica(self, lista):
        self.dico=self[lista]
        
    def DF_categorica(self, lista):
        self.cate=self[lista]   



    def limpiar_aux(self):
        
        try:
            df_nuevo=pd.DataFrame
            aux1=list(self.dico.columns)
            aux=[]
            df_nuevo=pd.get_dummies(self.df, columns=aux1)
            
            for columna in df_nuevo.columns:
                for variables in list(self.dico.columns):
                    if variables in columna:
                        aux.append(columna)
                    
            self.dummy=df_nuevo[aux]
            self[aux]=df_nuevo[aux]

            # self.df=self.drop(columns=var, axis='columns')
            # self.df= self[self.columns.difference(self.dico.columns)]
            
            print("********************** self.dummy ************\n")
            print(self.dummy)
            print("\n********************** self.df o todas_las_col() ************\n")
            print(self.df)

        except:
            print("---------------------- ERROR -----------------")



    def limpiar_dummys(self):

        b=False
        lista=list(self.dico.columns)
        for ind, i in enumerate(lista):
                if (ind+1<len(lista)):
                    if( (i in lista [ind+1]) ):
                        b=True
                        break
        if b:
            nombres_nuevos=[]
            if len(lista)>2:
                for ind, i in enumerate(lista):
                    if (ind+1<len(lista)):
                        if( (i in lista [ind+1]) ):
                            nombres_nuevos.append(i.upper())
                        else:
                            nombres_nuevos.append(i)
                    else:
                        nombres_nuevos.append(i)
                        
            aux_df=self.df

            for i,j in zip(lista,nombres_nuevos):
                aux_df.rename(columns={i:j},inplace=True)
                
            self.df=aux_df
            self.dico.columns=nombres_nuevos
            
            self.limpiar_aux()
        else: 
            self.limpiar_aux()



    def estadistica_descriptiva_cuantis(self):

        print("----------------------------------------------------------------------------------------------------\nDESCRIPCIÓN")
        print (self.cuanti.describe())
        print("\n")
        print("----------------------------------------------------------------------------------------------------\nCUARTILES")
        print (self.cuanti.quantile([0.05,0.25,0.5,0.75,0.95]))
        print("\n")
        print("----------------------------------------------------------------------------------------------------\n")
        print("\n")
        print("----------------------------------------------------------------------------------------------------\n")


        aux1=self.dico.columns
        aux2=self.cate.columns
        aux=self.cuanti.columns

        # df_auxiliar = self.groupby('sexo').apply(lambda x: pd.Series(shapiro(x), index=['W','P'])).reset_index()
        # print(df_auxiliar)
                
        for a in list(aux1.values):
            
            for b in list(aux.values):
                
                print("++++++++++++++++++++++++++++  "+a+" y "+b+"  ++++++++++++++++++++++++++\n")
                agrupado=self.groupby(a)[b]
                titulo=f"Agrupado por {a} y por {b}"
                print(titulo)
                print(agrupado.describe().reset_index())
                # df.groupby(['cat1', 'cat2'])['purchases','sales'].apply(stats.shapiro)
                print("////////////////////////// TEST DE SHAPIRO ////////////////////////////")
                aux_shapiro=(agrupado.apply(stats.shapiro))
                print(aux_shapiro)
        
                
                print("\n")
                print("----------------------------------------------------------------------------------------------------\n")


    def estadistica_descriptiva_cualis(self):

        print("\n--------------------- Variables dico ---------------------")
        print("\n")
        for i in self.dico.columns:
            print(f"...........Frecuencia variable {i} ....................")
            print(self[i].value_counts()/(self[i].count()))
            print("\n")

        print("\n-------------------- Variables categoricas --------------------")
        print("\n")
        for i in self.cate.columns:
            print(f"...........Frecuencia variable {i} ....................")
            print(self[i].value_counts()/(self[i].count()))
            print("\n")
        print("\n\n")

        # crosstab variables cualis con cate
        aux=list(self.cate.columns)

        a=0
        for i in aux:
            a=a+1
            if a<len(aux)/2:
                b=0
                for j in aux[:-1]:
                    b=b+1
                    if b > a:
                        print(f"*************** TABAL DE VARIABLES CATEGORICAS {i} y {j} *********************\n ")
                        tab = pd.crosstab (index=self[i], columns=self[j])
                        x=(tab/tab.sum())
                        print(tab)
                        print("\n")
                        print(f"/////////////////// EN PROPORCION //////////////////\n")
                        print(x)
                        print("\n\n")


    def anova(self):

        aux_cate=list(self.cate.columns)
        aux_cuati=list(self.cuanti.columns)

        for i in aux_cate:
            for j in aux_cuati:
                try:
                    print(f"\n----------- ANOVA Categoria {i} y variable continua {j} ----------\n")
                    model = ols(f"{j} ~ {i}", data=self).fit()
                    a=sm.stats.anova_lm(model, typ=2)
                    print(a)
                except:
                    print(f"\n - - - - - Fallo en variable {i} y {j} - - - - - - \n")
                    continue




    def Chi(self):

        aux_dico=list(self.dico.columns)

        if len(aux_dico)>1:
            for ind, i in enumerate(aux_dico):
                for j in range(ind+1,len(aux_dico)):
                    chi, p, dof, expected = stats.chi2_contingency(pd.crosstab(self[i],self[aux_dico[j]]), correction=False)
                    print(f"\n-------------- Chi2 entre {i} y {aux_dico[j]} ----------------")
                    print(f"p: {p} \n") 
        else:
            print("******************** No suficientes argumentos ********************")


    def t_test_aux(self, columns):
        results = []
        for i, col1 in enumerate(columns[:-1]):
            for col2 in columns[i+1:]:
                t, p = stats.ttest_ind(self[col1].dropna(), self[col2].dropna(), equal_var=False)
                # results.append((col1, col2, t, p))
                if p < 0.05:
                    print( f"+++++ Variable{col1}, variable 2 {col2} con p de: \033[1m{p}\033[0m  Se RECHAZA H0 ++++") 
                else:
                    print( f"+++++ Variable{col1}, variable 2 {col2}  con p de: {p} SE ACEPTA H0 ++++") 
    
    def wilcoxon_test_aux(self,col1, col2):
        if (col1== col2).all():
            print ("\nLas coluimnas son iguales\n")
        res = stats.wilcoxon(col1, col2)
        if res.pvalue < 0.05:
            print(f"Reject null hypothesis. Significant difference  (p-value={res.pvalue:.4f})")
        else:
            print (f"Fail to reject null hypothesis. No significant difference (p-value={res.pvalue:.4f})")

    def wilconxon(self, lista):
        # lista=[grupo, var]
        a,b=self.agrupar(lista)
        print(f"\n- Variable: {lista[1]}, Grupo: {lista[0]}")
        self.wilcoxon_test_aux(a, b)

    def agrupar (self, lista):
        groupby_col=lista[0]
        col=lista[1]
        valor=self[groupby_col].unique()
        group= self.where(self[groupby_col]== valor[0])[col]
        group2= self.where(self[groupby_col]== valor[1])[col] 
        return group,group2

    # def t_test_groupby_one_col(self, col, groupby_col):
        
    #     group= self.where(self[groupby_col]== self[groupby_col][0]).dropna()[col]
    #     group2= self.where(self[groupby_col]== self[groupby_col][1]).dropna()[col]
    #     t, p = stats.ttest_ind(group, group2, equal_var=False)
    #     print( col, groupby_col,p) 

    def t_test_all(self):
        aux=list(self.cuanti.columns)
        aux2=list(self.dico.columns)
        self.t_test_aux(self.normal_cuatis) #aqui ya hace todas las cuantis entre ellas faltan los grupos
        for i in self.normal_grupos_dico:
            a,b=self.agrupar(i)
            t, p = stats.ttest_ind(a.dropna(), b.dropna(), equal_var=False)
            if p < 0.05:
                print( f"+++++ Variable{i[1]}, Agrupado por {i[0]} con p de: \033[1m{p}\033[0m  Se RECHAZA H0 ++++") 
            else:
                print( f"+++++ Variable{i[1]}, Agrupado por {i[0]} con p de: {p} SE ACEPTA H0 ++++") 
    # df_prueba.groupby('sexo').apply(lambda df: stats.ttest_ind(df['Datos_D'].dropna(), df['Datos_E'].dropna())[1])


    def plot_confidence_interval(self, col, confidence_level= 0.95):
        data = self[col].to_numpy()
        n = len(data)
        mean =self[col].mean(axis=0)
        # std_error = stats.sem( self[col].dropna())
        std_error = self[col].dropna().std()
        lower_bound = stats.t.ppf(0.025, n - 1, loc = mean, scale = std_error)  # =>  99.23452406698323
        upper_bound = stats.t.ppf(0.975, n - 1, loc = mean, scale = std_error)
        # h = std_error * stats.t.ppf((1 + confidence_level) / 2, n - 1)
        
        # lower_bound = mean - h
        # upper_bound = mean + h
        # plt.hist(data, bins=30, edgecolor='black', alpha=0.5)
        # plt.axvspan(lower_bound, upper_bound, color='gray', alpha=0.2, label=f'{confidence_level * 100}% Confidence Interval')
        # plt.axvline(x=mean, color='red', label='Sample Mean')
        # plt.legend()

        fig, ax = plt.subplots()
        ax.hist(data, bins=30, edgecolor='black', alpha=0.5)
        ax.axvline(x=mean, color='red', label='Sample Mean')
        ax.axvspan(lower_bound, upper_bound, color='grey', alpha=0.5, label=f'{confidence_level * 100}% Confidence Interval')
        ax.annotate(
            f'lower_bound:\n {lower_bound:.2f}',
            xy=(lower_bound, 0), xytext=(lower_bound-0.5, 50)
        )
        ax.annotate(
            f'upper_bound:\n  {upper_bound:.2f}',
            xy=(upper_bound, 0), xytext=(upper_bound-0.5, 50)
        )
        ax.legend()
        
        plt.show()


    def plot_normailidad(self):
        aux=self.cuanti.columns
        for i in aux:
            stats.probplot(self[i], dist="norm", plot=plt)
            plt.title("Probability Plot - " )
            plt.show()
    


    def plot_bigotes(self):

        aux1=self.dico.columns
        aux2=self.cate.columns
        aux=self.cuanti.columns
        print("\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")

        print("-------------- Graficas de bigotes cualitativas-------------------")
        # fig = plt.figure(figsize=(12, 8))
        
        (self.cuanti).plot(kind='box', title='Variables cuantitativas',figsize=(12, 8))
        plt.show()
        

        print("-------------- Graficas de bigotes por dicotomicas-------------------")   
        
        for a in aux1:

            # fig = plt.figure(figsize=(12, 8))
            self.boxplot(column=list(aux.values), by=a,figsize=(12, 8))
            plt.tight_layout() 
            plt.show()
        
        print("\n")
        print("----------------------------------------------------------------------------------------------------\n")
        
        print("-------------- Graficas de bigotes por categoricas-------------------") 

        for a in aux2:
            # fig = plt.figure(figsize=(12, 8))
            ax= self.boxplot(column=list(aux.values), by=a, figsize=(12, 8))
            # ax = sns.swarmplot(column=list(aux.values), by=a,data=self, color='#7d0013')
            plt.tight_layout() 
            plt.show()
        print("\n")
        print("----------------------------------------------------------------------------------------------------\n")

        
        print("\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")



    def plot_corr(self):

        print("\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
        print("----------------------------------------------------------------------------------------------------\n")
        
        print("-------------- MATRIZ DE CORRELACIONES ENTRE CUANTITATIVAS -------------------\n") 

        fig = plt.figure(figsize=(12, 8))
        matrix = self.cuanti.corr().round(2)
        mask = np.triu(np.ones_like(matrix, dtype=bool))
        sns.heatmap(matrix, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag', mask=mask)  
        plt.show()

        print("----------------------------------------------------------------------------------------------------\n")

        print("\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")


    def plot_barras(self):
        aux=self.cuanti.columns

        print("\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
        print("----------------------------------------------------------------------------------------------------\n")

        print("-------------- GRAFICA DE BARRAS DE TODAS LAS CUANTITATIVAS -------------------\n") 
        # fig = plt.figure(figsize=(15, 20))
        self.cuanti.plot.bar(figsize=(18, 8))
        plt.show()

        print("-------------- GRAFICA DE BARRAS CON DISTRIBUCIÓN DE DENSIDAD DE CADA CUANTITATIVA  -------------------\n") 
        for i in list(aux.values):
            fig = plt.figure(figsize=(12, 8))
            print(f"\n.............. GRAFICA DE BARRAS  DE {i} ............\n") 
            ax=self[i].plot.hist(density=True)
            self[i].plot.density(ax=ax)
            plt.show()

        print("----------------------------------------------------------------------------------------------------\n")    
        print("\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")


    def todos_plots(self):

        self.plot_bigotes()
        self.plot_corr()
        self.plot_barras()
        self.violines()
        
        

    def violines(self):

        aux1=self.dico.columns
        aux2=self.cate.columns
        aux=self.cuanti.columns

        print("--------------  GRAFICA DE VIOLINES  -------------------\n") 
        sns.set(style="whitegrid")
        for i in aux2:
            for j in aux:
                ax= sns.violinplot(x=self[i], y=self[j], palette="Set2", split=True, inner="quartile",scale="count")
                plt.show()

        print("\n\n/////////-------------- GRAFICA DE VIOLINES POR DICOTOMICAS -------------------/////////////\n") 
        
        for i in aux2:
            for j in aux:
                for k in aux1:
                    ax= sns.violinplot(x=self[i], y=self[j], hue=self[k],palette="Set2", split=True, inner="quartile",scale="count")
                    plt.show()


    def cross_var_cualis_con_ciantis(self):

        aux=list(self.cate.columns)
        aux_cuati=list(self.cuanti.columns)

        for k in aux_cuati:
            a=0
            for i in aux:
                a=a+1
                if a<len(aux)/2:
                    b=0
                    for j in aux[:-1]:
                        b=b+1
                        if b > a:
                            print(f"\n\n*************** TABAL DE VARIABLES CATEGORICAS {i} y {j} con valores de {k} MEDIA *********************\n ")
                            tab = pd.crosstab (index=self[i], columns=self[j],values=self[k],aggfunc=np.mean)
                            print(tab)
                            print("\n\n")

    def nulos(self):
        aux_df=list(self.cuanti.columns)
        self.inputado=self.df
        for i in aux_df:
            nulos=self[i].isna().sum()
            total=len(self[i])
            porcentaje=nulos/total
            if ((nulos>0)):
                percen=self[i].quantile([0.2,0.8]).to_list()
                self[i]=self[i].apply(lambda x: ( random.randint ( round(percen[0]) , round(percen[1]) )) if pd.isna(x) else x )
                print(f"\n- Se han inputado {nulos} nulos a la variable {i} (tenía porcentaje de nulos de: {porcentaje}) \n")
            elif (porcentaje>self.porcentaje_nulos_permitido):
                print(f"\n - No se ha podido inputar a la variable {i} porque el porcentaje de nulos era de {porcentaje}\n")
                

    def normalidad(self):
        
        DataF=self.df
        aux1=self.dico.columns
        aux2=self.cate.columns
        aux=self.cuanti.columns
                
        for b in list(aux.values):
            aux_shapiro=(stats.shapiro(DataF[b]))
            if(aux_shapiro.pvalue<0.05):
                print("////////////////////////// TEST DE SHAPIRO CUANTITATIVAS ////////////////////////////")
                print("++++++++++++++++++++++++++++  "+ b +"  ++++++++++++++++++++++++++\n")
                titulo=f"Variable cuantitativa {b} y test Shapiro < 0.05"
                print(titulo)
                print(aux_shapiro)
                print("\n")
                print("----------------------------------------------------------------------------------------------------\n")
                self.normal_cuatis.append(b)

        for a in list(aux1.values):
            for b in list(aux.values):
                    agrupado=DataF.groupby(a)[b]
                    try:
                        aux_shapiro=(agrupado.apply(stats.shapiro))
                        for h in aux_shapiro:
                            if(h.pvalue<0.05):
                                print("////////////////////////// TEST DE SHAPIRO DICOTOMICAS ////////////////////////////")
                                print("++++++++++++++++++++++++++++  "+a+" y "+b+"  ++++++++++++++++++++++++++\n")
                                titulo=f"Agrupado por {a} y por {b} y test Shapiro < 0.05"
                                print(titulo)
                                print(aux_shapiro)
                                print("\n")
                                print("----------------------------------------------------------------------------------------------------\n")
                                self.normal_grupos_dico.append([a,b])
                    except:
                        continue 

        for a in list(aux2.values):
            for b in list(aux.values):
                    agrupado=DataF.groupby([a])[b]
                    try:
                        aux_shapiro=(agrupado.apply(stats.shapiro))
                        for h in aux_shapiro:
                            if(h.pvalue<0.05):
                                print("////////////////////////// TEST DE SHAPIRO CATEGORICAS ////////////////////////////")
                                print("++++++++++++++++++++++++++++  "+a+" y "+b+"  ++++++++++++++++++++++++++\n")
                                titulo=f"Agrupado por {a} y por {b} y test Shapiro < 0.05"
                                print(titulo)
                                print(h)
                                print("\n")
                                print("----------------------------------------------------------------------------------------------------\n")
                                self.normal_grupos_cate.append([a,b])
                    except:
                        continue 

        self.normal_grupos_dico=[i for n, i in enumerate(self.normal_grupos_dico) if i not in self.normal_grupos_dico[:n]]
        self.normal_grupos_cate=[i for n, i in enumerate(self.normal_grupos_cate) if i not in self.normal_grupos_cate[:n]]
        
    def detec_outlaiers(self):
        aux=list(self.cuanti.columns)
        aux_DF=self.cuanti
        for i in aux:
            z = np.abs(stats.zscore(aux_DF[i]))
            print(z)
    
    def seleccionar_distribuciones(self,familia='realall', verbose=False):
        '''
        Parameters
        ----------
        familia : {'realall', 'realline', 'realplus', 'real0to1', 'discreta'}
            realall: distribuciones de la familia `realline` + `realplus`
            realline: distribuciones continuas en el dominio (-inf, +inf)
            realplus: distribuciones continuas en el dominio [0, +inf)
            real0to1: distribuciones continuas en el dominio [0,1]
            discreta: distribuciones discretas
            
        verbose : bool
            Si se muestra información de las distribuciones seleccionadas
            (the default `False`)
        '''
    
        distribuciones = [getattr(stats,d) for d in dir(stats) \
                        if isinstance(getattr(stats,d), (stats.rv_continuous, stats.rv_discrete))]
        
        exclusiones = ['levy_stable', 'vonmises']
        distribuciones = [dist for dist in distribuciones if dist.name not in exclusiones]
                
        dominios = {
            'realall' : [-np.inf, np.inf],
            'realline': [np.inf,np.inf],
            'realplus': [0, np.inf],
            'real0to1': [0, 1], 
            'discreta': [None, None],
        }

        distribucion = []
        tipo = []
        dominio_inf = []
        dominio_sup = []

        for dist in distribuciones:
            distribucion.append(dist.name)
            tipo.append(np.where(isinstance(dist, stats.rv_continuous), 'continua', 'discreta'))
            dominio_inf.append(dist.a)
            dominio_sup.append(dist.b)
        
        info_distribuciones = pd.DataFrame({
                                'distribucion': distribucion,
                                'tipo': tipo,
                                'dominio_inf': dominio_inf,
                                'dominio_sup': dominio_sup
                            })

        info_distribuciones = info_distribuciones \
                            .sort_values(by=['dominio_inf', 'dominio_sup'])\
                            .reset_index(drop=True)
        
        if familia in ['realall', 'realline', 'realplus', 'real0to1']:
            info_distribuciones = info_distribuciones[info_distribuciones['tipo']=='continua']
            condicion = (info_distribuciones['dominio_inf'] == dominios[familia][0]) & \
                        (info_distribuciones['dominio_sup'] == dominios[familia][1]) 
            info_distribuciones = info_distribuciones[condicion].reset_index(drop=True)
            
        if familia in ['discreta']:
            info_distribuciones = info_distribuciones[info_distribuciones['tipo']=='discreta']
            
        seleccion = [dist for dist in distribuciones \
                    if dist.name in info_distribuciones['distribucion'].values]
        
        
        if verbose:
            print("---------------------------------------------------")
            print("       Distribuciones seleccionadas                ")
            print("---------------------------------------------------")
            with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
                print(info_distribuciones)
        
        return seleccion


    def plot_multiple_distribuciones(self, nombre_distribuciones):

        aux=list(self.cuanti.columns)
        fig, ax = plt.subplots(figsize=(15,15))

        for i in aux:
            x=self[i]
            if ax is None:
                fig, ax = plt.subplots(figsize=(7,4))
                
            ax.hist(x=x, density=True, bins=30, color="#3182bd", alpha=0.5)
            ax.plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)
            ax.set_title('Ajuste distribuciones')
            ax.set_xlabel('x')
            ax.set_ylabel('Densidad de probabilidad')
            
            for nombre in nombre_distribuciones:
                
                distribucion = getattr(stats, nombre)

                parametros = distribucion.fit(data=x)

                nombre_parametros = [p for p in inspect.signature(distribucion._pdf).parameters \
                                    if not p=='x'] + ["loc","scale"]
                parametros_dict = dict(zip(nombre_parametros, parametros))

                log_likelihood = distribucion.logpdf(x, *parametros).sum()

                aic = -2 * log_likelihood + 2 * len(parametros)
                bic = -2 * log_likelihood + np.log(x.shape[0]) * len(parametros)

                x_hat = np.linspace(min(x), max(x), num=100)
                y_hat = distribucion.pdf(x_hat, *parametros)
                ax.plot(x_hat, y_hat, linewidth=2, label=distribucion.name)
            
            ax.legend()
            plt.show()
        


    def fit_discrete(self,datos):

        # self.discreta

        mean = datos.mean()
        var = datos.var()
        likelihoods = {}  
        log_likelihoods = {}

        p = 1 - mean / var  
        r = (1-p) * mean / p



        log_likelihoods['nbinom'] = datos.map(lambda val: stats.nbinom.logpmf(val, r, p)).sum()

        lambda_ = mean

        log_likelihoods['poisson'] = datos.map(lambda val: stats.poisson.logpmf(val, lambda_)).sum()


        best_fit = max(log_likelihoods, key=lambda x: log_likelihoods[x])
        print("**** Best fit between poisson and nbinorm :", best_fit)
        

    
        plt.hist(datos, bins=int(np.max(datos)), density=True, alpha=0.5)

        mean = datos.mean()
        var = datos.var()


        def loss_function_poisson(params, datos_in):

            mu = params[0]

            loss = 0

            for i in range(len(datos_in)):

                loglikelihood = stats.poisson.logpmf(datos_in[i], mu)

                loss_to_add = -loglikelihood

                loss += loss_to_add

            return(loss)




        params0 = np.array([20])
        minimum = stats2.optimize.fmin(loss_function_poisson, params0, args=(datos,))

        mu_fit = minimum[0]

        print("***********  The best mu_fit is:  ",  mu_fit)
        print("\n")

        x = list(range(int(np.min(datos)), int(np.max(datos))+1))
        plt.scatter(x, stats.poisson.pmf(x, mu_fit),color="red")
        plt.show()   

        print("\n\n Otras variables discretas:  ",  self.discreta)


    def comparar_distribuciones_caunti_cont(self, ordenar='aic', verbose=False):

            '''
            resultados: data.frame
                distribucion: nombre de la distribución.
                log_likelihood: logaritmo del likelihood del ajuste.
                aic: métrica AIC.
                bic: métrica BIC.
                n_parametros: número de parámetros de la distribución de la distribución.
                parametros: parámetros del tras el ajuste
                
            Raises
            ------
            Exception
                Si `familia` es distinto de 'realall', 'realline', 'realplus', 'real0to1',
                o 'discreta'.
                
            Notes
            -----
            '''
            aux=list(self.cuanti.columns)
            
            for i in aux:
                print(f"\n ******************** Variable: {i} ******************** \n")
                x=self[i]
                distribuciones = self.seleccionar_distribuciones(familia='realall',verbose=verbose)
                distribucion_ = []
                log_likelihood_= []
                aic_ = []
                bic_ = []
                n_parametros_ = []
                parametros_ = []
                
                for j, distribucion in enumerate(distribuciones):
                    
                    # print(f"{j+1}/{len(distribuciones)} Ajustando distribución: {distribucion.name}")
                    
                    try:
                        parametros = distribucion.fit(data=x)
                        nombre_parametros = [p for p in inspect.signature(distribucion._pdf).parameters \
                                            if not p=='x'] + ["loc","scale"]
                        parametros_dict = dict(zip(nombre_parametros, parametros))
                        log_likelihood = distribucion.logpdf(x, *parametros).sum()
                        aic = -2 * log_likelihood + 2 * len(parametros)
                        bic = -2 * log_likelihood + np.log(x.shape[0]) * len(parametros)
                        
                        distribucion_.append(distribucion.name)
                        log_likelihood_.append(log_likelihood)
                        aic_.append(aic)
                        bic_.append(bic)
                        n_parametros_.append(len(parametros))
                        parametros_.append(parametros_dict)
                        
                        resultados = pd.DataFrame({
                                        'distribucion': distribucion_,
                                        'log_likelihood': log_likelihood_,
                                        'aic': aic_,
                                        'bic': bic_,
                                        'n_parametros': n_parametros_,
                                        'parametros': parametros_,
                            
                                    })
                        
                        resultados = resultados.sort_values(by=ordenar).reset_index(drop=True)

                        
                        
                    except Exception as e:
                        print(f"Error al tratar de ajustar la distribución {distribucion.name}")
                        print(e)
                        print("")

                nombre_distribuciones=resultados['distribucion'][:5]
                fig, ax = plt.subplots(figsize=(7,4))
                
                
                ax.hist(x=x, density=True, bins=30, color="#3182bd", alpha=0.5)
                ax.plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)
                ax.set_title('Ajuste distribuciones')
                ax.set_xlabel('x')
                ax.set_ylabel('Densidad de probabilidad')
                
                for nombre in nombre_distribuciones:
                    
                    distribucion = getattr(stats, nombre)

                    parametros = distribucion.fit(data=x)

                    nombre_parametros = [p for p in inspect.signature(distribucion._pdf).parameters \
                                        if not p=='x'] + ["loc","scale"]
                    parametros_dict = dict(zip(nombre_parametros, parametros))

                    log_likelihood = distribucion.logpdf(x, *parametros).sum()

                    aic = -2 * log_likelihood + 2 * len(parametros)
                    bic = -2 * log_likelihood + np.log(x.shape[0]) * len(parametros)

                    x_hat = np.linspace(min(x), max(x), num=100)
                    y_hat = distribucion.pdf(x_hat, *parametros)
                    ax.plot(x_hat, y_hat, linewidth=2, label=distribucion.name)
            
                ax.legend()
                plt.show()

                print("\n")
                print(resultados.head(5))    
                print("\n------------------------------------------------------------------\n")

    def remove_outliers(self, k=1.5):
        aux=list(self.cuanti.columns)
        for column in aux:
            print(f"\n\n                    <<<<<<<<<<<<<<<<<<<<<<<< {column} >>>>>>>>>>>>>>>>>>>>>>>>\n\n")
            self.plot_outliers2(column, k=1.5)
            q1, q3 = self[column].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound = q1 - (k * iqr)
            upper_bound = q3 + (k * iqr)
            self.loc[(self[column] < lower_bound) | (self[column] > upper_bound), column] = None    
        self.nulos()
        self.outliers_hecho=False


    def plot_outliers(self, column, k=1.5):
        
        q1, q3 = self[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - (k * iqr)
        upper_bound = q3 + (k * iqr)
        
        fig, ax = plt.subplots()
        ax.scatter(self.index, self[column], color='blue', label='inlier')
        ax.scatter(self[(self[column] < lower_bound) | (self[column] > upper_bound)].index,
                self[(self[column] < lower_bound) | (self[column] > upper_bound)][column],
                color='red', label='outlier')
        ax.axhline(lower_bound, color='gray', linestyle='--')
        ax.axhline(upper_bound, color='gray', linestyle='--')
        plt.legend()
        plt.show() 

    
    def plot_outliers2(df, column, k=1.5):
        q1, q3 = df[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - (k * iqr)
        upper_bound = q3 + (k * iqr)
        
        fig, ax = plt.subplots()
        ax.plot(df.index, df[column], color='blue')
        ax.scatter(df[df[column].isnull()].index,
                df[df[column].isnull()][column],
                color='red', marker='x')
        ax.axhline(lower_bound, color='red', linestyle='--')
        ax.axhline(upper_bound, color='red', linestyle='--')
        plt.show()
        
    def plot_xy_data(df, x_column, y_column):
        fig, ax = plt.subplots()
        ax.scatter(df[x_column], df[y_column])
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.show()


    def reg_lineal(self, predictores, OUTCOME):
        aux1=set(predictores)
        aux2=set(list(self.cuali.columns))
        

        if (aux1.intersection(aux2)): 
            print("Tienes alguna variable cualitativa en los predictores")
        elif(OUTCOME in list(self.cuali.columns)):
            print("OUTCOME es cualitativa") 
        #No funciona con NA ni con distinta longitud dentro de los DF
        if (self.outliers_hecho):
            try:
                self.remove_outliers()
                self.predicotres=self[predictores]
                self.outcome=self[OUTCOME]
                ej_lm=LinearRegression()
                ej_lm.fit(self.predicotres,self.outcome)

                for name, coef in zip(predictores,ej_lm.coef_):
                    print(f"{name}: {coef}")

                fitted= ej_lm.predict(self.predicotres)
                RMSE= np.sqrt(mean_squared_error(self.outcome,fitted))
                r2= r2_score(self.outcome,fitted) 

                # RMSE es como el accuracy del modelo (es practicamente igual al RSE)
                print(f"- RMSE: {RMSE:.0f}")

                # coeficiente de determinación:  0-1 proporción de varianza en los datos
                # que estan contabilizados en el modelo
                print(f"- R2: {r2:.4f}")
                model=sm.OLS(self.outcome,self.predicotres.assign(const=1) )
                resul=model.fit()
                print("\n - RESUMEN \n")
                print( resul.summary())
                return ej_lm
            except:
                print("Puede que haya columans con distinta longitud")

        elif (self[predictores].isna().any().any()):
            print("HAY VALORES NULOS EN LAS COLUMNAS Y YA HAS HECHO LA FUNCIÓN DE OUTLIERS")

        else :
            try:
                self.predicotres=self[predictores]
                self.outcome=self[OUTCOME]
                ej_lm=LinearRegression()
                ej_lm.fit(self.predicotres,self.outcome)

                for name, coef in zip(predictores,ej_lm.coef_):
                    print(f"{name}: {coef}")

                fitted= ej_lm.predict(self.predicotres)
                RMSE= np.sqrt(mean_squared_error(self.outcome,fitted))
                r2= r2_score(self.outcome,fitted) 

                # RMSE es como el accuracy del modelo (es practicamente igual al RSE)
                print(f"- RMSE: {RMSE:.0f}")

                # coeficiente de determinación:  0-1 proporción de varianza en los datos
                # que estan contabilizados en el modelo
                print(f"- R2: {r2:.4f}")
                model=sm.OLS(self.outcome,self.predicotres.assign(const=1) )
                resul=model.fit()
                print("\n ------------------------- RESUMEN --------------------------- \n")
                print( resul.summary())
                return ej_lm
            except:
                print("Puede que haya columans con distinta longitud")


    def forward_selected(self):
        """Linear model designed by forward selection.

        Parameters:
        -----------
        data : pandas DataFrame with all possible predictors and response

        response: string, name of response column in data

        Returns:
        --------
        model: an "optimal" fitted statsmodels linear model
            with an intercept
            selected by forward selection
            evaluated by adjusted R-squared
        """
        data=pd.merge(self.predicotres, self.outcome,left_index=True, right_index=True)
        response=self.outcome.columns[0]

        remaining = set(data.columns)
        remaining.remove(response)
        selected = []
        current_score, best_new_score = 0.0, 0.0
        while remaining and current_score == best_new_score:
            scores_with_candidates = []
            for candidate in remaining:
                formula = "{} ~ {} + 1".format(response,
                                            ' + '.join(selected + [candidate]))
                score = ols(formula, data).fit().rsquared_adj
                scores_with_candidates.append((score, candidate))
            scores_with_candidates.sort()
            best_new_score, best_candidate = scores_with_candidates.pop()
            if current_score < best_new_score:
                remaining.remove(best_candidate)
                selected.append(best_candidate)
                current_score = best_new_score
        formula = "{} ~ {} + 1".format(response,
                                    ' + '.join(selected))
        model = ols(formula, data).fit()

        print (f"Formula: {model.model.formula}")
        print (f"Ajuste por R2: {model.rsquared_adj}")

        return model




    def weighted_regression(self, weights):
        X = self.predicotres.values
        y = self.outcome.values
        w = weights.values

        model = LinearRegression()
        model.fit(X, y, sample_weight=w)

        return model
    
    def codificar_catego(self,modelo,cate_var:str):
        self['residuos']=(self.outcome-modelo.predict(self.predicotres))
        self['residuos']
        self[cate_var]= self[cate_var]
        grupos1_aux= pd.merge(self[cate_var], self['residuos'],left_index=True, right_index=True)
        grupos1_agrupado=grupos1_aux.groupby([cate_var])

        summary_function = lambda x: {
            cate_var: x.iloc[0,0],
            'count': len(x),
            'residuo_medio': x.residuos.median()
        }
        
        group_summaries = grupos1_agrupado.apply(summary_function)
        final_df = pd.DataFrame([    *group_summaries])
        grupos1 = final_df.sort_values('residuo_medio')
        grupos1['cum_count']=np.cumsum(grupos1['count'])
        grupos1['Col_a_codificar_grupos']=pd.qcut(grupos1['cum_count'],5,labels=False,retbins=False)
        to_join= grupos1[[cate_var,'Col_a_codificar_grupos']].set_index(cate_var)

        self=self.join(to_join, on=cate_var)
        return (self[[cate_var,'Col_a_codificar_grupos']])
    
    def regre_con_interaccion_de_var(self,outcome,predictores,lista_predictores_condicionados):
        frase=outcome+" ~"
        frase_aux1=""
        frase_aux2=""
        aux=0
        for i, j in lista_predictores_condicionados:
            if aux==0:
                frase_aux1=i+"*"+j
            else:
                frase_aux1=frase_aux1+"+"+i+"*"+j
            aux=aux+1
        for i in predictores:
            frase_aux2=frase_aux2+"+"+i
        frase=frase+frase_aux1+frase_aux2
        print(f" Formula final: {frase} \n\n")
        model=smf.ols(formula= frase,data=self )
        results=model.fit()
        return results.summary()
    

    def regre_outliers(self, cate=None, grupo=None):

        if cate==None:
            ej_outliers=sm.OLS(self.outcome, self.predicotres.assign(conts=1))
            resul_1=ej_outliers.fit()

            influence=OLSInfluence(resul_1)
            sresiduals= influence.resid_studentized_internal

            outliers=self.loc[sresiduals.idxmin(), :]
            print("resultado", outliers[list(self.outcome.columns)])
            print(outliers[list(self.predicotres.columns)])


            print("puntos con alta influencia y distancia de Cooks mayor de 0.08")
            fig, ax=plt.subplots(figsize=(5,5))
            ax.axhline(-2.5, linestyle='--', color='C1')
            ax.axhline(2.5, linestyle='--', color='C1')
            ax.scatter(influence.hat_matrix_diag, 
                    influence.resid_studentized_internal,
                    s=1000*np.sqrt(influence.cooks_distance[0]), 
                    alpha=0.5)
            ax.set_xlabel('hat values')
            ax.set_ylabel('studentized residuals')
            plt.show()


            print("predictores vs residuos para ver heteroskedascity")
            fig, ax=plt.subplots(figsize=(5,5))
            df = pd.DataFrame({'fitted': resul_1.fittedvalues, 
                            'residuals': np.abs(resul_1.resid)})
            sns.regplot(x='fitted', y='residuals', data=df, scatter_kws={'alpha':0.25}, line_kws={'color': 'C1'}, lowess=True, ax=ax)
            ax.set_xlabel('predictos')
            ax.set_ylabel('abs(residuos)')
            plt.show()

        else:
            datos_agru=self.loc[df_prueba[cate]==grupo,]
            ej_outliers=sm.OLS(datos_agru[list(self.outcome.columns)], datos_agru[list(self.predicotres.columns)].assign(conts=1))
            resul_1=ej_outliers.fit()

            influence=OLSInfluence(resul_1)
            sresiduals= influence.resid_studentized_internal

            outliers=datos_agru.loc[sresiduals.idxmin(), :]
            print("resultado", outliers[list(self.outcome.columns)])
            print(outliers[list(self.predicotres.columns)])

            print("puntos con alta influencia y distancia de Cooks mayor de 0.08")
            fig, ax=plt.subplots(figsize=(5,5))
            ax.axhline(-2.5, linestyle='--', color='C1')
            ax.axhline(2.5, linestyle='--', color='C1')
            ax.scatter(influence.hat_matrix_diag, 
                    influence.resid_studentized_internal,
                    s=1000*np.sqrt(influence.cooks_distance[0]), 
                    alpha=0.5)
            ax.set_xlabel('hat values')
            ax.set_ylabel('studentized residuals')
            plt.show()

            print("predictores vs residuos para ver heteroskedascity")
            fig, ax=plt.subplots(figsize=(5,5))
            df = pd.DataFrame({'fitted': resul_1.fittedvalues, 
                            'residuals': np.abs(resul_1.resid)})
            sns.regplot(x='fitted', y='residuals', data=df, scatter_kws={'alpha':0.25}, line_kws={'color': 'C1'}, lowess=True, ax=ax)
            ax.set_xlabel('predictos')
            ax.set_ylabel('abs(residuos)')
            plt.show()

    def infl_residual_modelo(self, var_influ, cate=None, grupo=None):
        if cate==None:
            ej_outliers=sm.OLS(self.outcome, self.predicotres.assign(conts=1))
            resul_1=ej_outliers.fit()
            sm.graphics.plot_ccpr(resul_1,var_influ)
            fig = plt.figure(figsize=(12, 12))
            fig = sm.graphics.plot_ccpr_grid(resul_1, fig=fig)
        
        else:
            datos_agru=self.loc[df_prueba[cate]==grupo,]
            ej_outliers=sm.OLS(datos_agru[list(self.outcome.columns)], datos_agru[list(self.predicotres.columns)].assign(conts=1))
            resul_1=ej_outliers.fit()
            sm.graphics.plot_ccpr(resul_1,var_influ)
            fig = plt.figure(figsize=(12, 12))
            fig = sm.graphics.plot_ccpr_grid(resul_1, fig=fig)

    def regre_poly(self,variables_exp,expo,cate=None, grupo=None,verbose=False):

        if cate==None:

            out=list(self.outcome.columns)
            predic=list(self.predicotres.columns)

            frase=out[0]+" ~ "
            variables_no_exp = [element for element in predic if element not in variables_exp]
            frase_expo=""
            frase_no_expo=""

            for i,j in zip(variables_exp,expo):
                frase_expo=frase_expo + f"np.power({i}, {j}) + " 

            for indice,i in enumerate(variables_no_exp):
                if indice == len(variables_no_exp)-1:
                    frase_no_expo=frase_no_expo+i    
                else: 
                    frase_no_expo=frase_no_expo+i+ "+"   

            frase=frase+frase_expo+frase_no_expo
            print(frase)
            
            model_poly = smf.ols(formula=frase, data=self)
            result_poly = model_poly.fit()
            if (verbose):
             print(result_poly.summary())

        else:

            datos_agru=self.loc[df_prueba[cate]==grupo,]

            out=list(self.outcome.columns)
            predic=list(self.predicotres.columns)

            frase=out[0]+" ~ "
            variables_no_exp = [element for element in predic if element not in variables_exp]
            frase_expo=""
            frase_no_expo=""

            for i,j in zip(variables_exp,expo):
                frase_expo=frase_expo + f"np.power({i}, {j}) + " 

            for indice,i in enumerate(variables_no_exp):
                if indice == len(variables_no_exp)-1:
                    frase_no_expo=frase_no_expo+i    
                else: 
                    frase_no_expo=frase_no_expo+i+ "+"   

            frase=frase+frase_expo+frase_no_expo
            print(frase)
            
            model_poly = smf.ols(formula=frase, data=datos_agru)
            result_poly = model_poly.fit()
            if (verbose):
                print(result_poly.summary())

        return result_poly
        

    def partialResidualPlot(self, model, feature):
        df= pd.merge(self.predicotres, self.outcome, left_index=True, right_index=True)
        outcome= list(self.outcome.columns)

        y_pred = model.predict(df)
        copy_df = df.copy()
        for c in copy_df.columns:
            if c == feature:
                continue
            copy_df[c] = 0.0
        feature_prediction = model.predict(copy_df)
        
        
        residual=df[outcome].values - y_pred.values
        results = pd.DataFrame({
            'feature': df[feature].values,
            'residual': residual[0],
            'ypartial': feature_prediction.values - model.params[0],
        })

        results = results.sort_values(by=['feature'])
        smoothed = sm.nonparametric.lowess(results.ypartial, results.feature, frac=1/3)
        
        fig, ax = plt.subplots(figsize=(5, 5))

        ax.scatter(results.feature, results.ypartial + results.residual)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color='gray')
        ax.plot(results.feature, results.ypartial, color='black')
        ax.set_xlabel(feature)
        ax.set_ylabel(f'Residual + {feature} contribution')
        plt.tight_layout()
        plt.show()
        
    
    def plot_partial_residuals_poly(self,variables_exp,expo,variable ,cate=None, grupo=None):
        model=self.regre_poly(variables_exp,expo,cate, grupo)
        self.partialResidualPlot(model,variable)

    
    def clasificador_bayes(self,predictores,outcome,new):
        X =self[predictores]
        y = self[outcome]

        naive_model = MultinomialNB(alpha=0.01, fit_prior=True)
        naive_model = MultinomialNB(alpha=1e-10, fit_prior=False)
        naive_model.fit(X, y)

        print("Input en el modelo bayesiano: ")
        print(new)
        print("\n")

        print('Clase más probable: ', naive_model.predict(new)[0])

        probabilities = pd.DataFrame(naive_model.predict_proba(new),columns=naive_model.classes_)
        print('Probabilidades de cada clase:',)
        print(probabilities)

    def accuracy_bayes(self,predictores,outcome):
        X =self[predictores]
        y = self[outcome]

        X_train, X_test, y_train, y_test = train_test_split(X, y)

        modelo_gausian = GaussianNB()
        
        modelo_gausian .fit(X_train, y_train)

        y_pred = modelo_gausian .predict(X_test)

        precision_gausian  = accuracy_score(y_test, y_pred)

        print(f"La precisión del modelo gaussiano de bayes es: {precision_gausian }")

        modelo_multino=MultinomialNB()

        modelo_multino.fit(X_train, y_train)

        y_pred = modelo_multino.predict(X_test)

        precision_multino = accuracy_score(y_test, y_pred)

        print(f"La precisión del modelo multinomial de bayes es: {precision_multino}")

    def predict_lda(self,predictors,outcome):
        X=self[predictors]
        y=self[outcome]

        modelo_lda = LinearDiscriminantAnalysis()
        modelo_lda.fit(X, y)
        y_pred = modelo_lda.predict(X)

        accuracy = accuracy_score(y, y_pred)
        print("Accuracy modelo LDA:", accuracy)

    def GLM_datos(self,predictors,outcome):
        if (outcome in list(self.dummy.columns)):
            y= self[outcome]
            X= self[predictors]

            logit_reg_sm = sm.GLM(y, X.assign(const=1), 
                                family=sm.families.Binomial())
            logit_result = logit_reg_sm.fit()
            print(logit_result.summary())
        else:
            print(" El outcome no está bien escrito prueba con alguno de estos:\n")
            print(list(self.dummy.columns))

    def GLM_datos_formula(self,predictors,outcome,formula):
        if (outcome in list(self.dummy.columns)):
            y= self[outcome]
            X= self[predictors]
            data=pd.merge(y, X,left_index=True, right_index=True)
            model = glm(formula=formula, data=data, family=sm.families.Binomial())
            results = model.fit()
            print(results.summary())
        else:
            print(" El outcome no está bien escrito prueba con alguno de estos:\n")
            print(list(self.dummy.columns))

    def mat_conf(self,predictors,outcome):
        if (outcome in list(self.dummy.columns)):
            y= self[outcome]
            X= self[predictors]
            logit_reg = LogisticRegression(penalty='l2', C=1e42, solver='liblinear')
            logit_reg.fit(X, y)

            print(' - Intercept ', logit_reg.intercept_[0])
            print(' - Classes', logit_reg.classes_)
            pd.DataFrame({'coeff': logit_reg.coef_[0]}, 
                        index=X.columns)

            # Confusion matrix
            pred = logit_reg.predict(X)
            pred_y = logit_reg.predict(X) == "0"
            true_y = y == "0"
            true_pos = true_y & pred_y
            true_neg = ~true_y & ~pred_y
            false_pos = ~true_y & pred_y
            false_neg = true_y & ~pred_y

            conf_mat = pd.DataFrame([[np.sum(true_pos), np.sum(false_neg)], [np.sum(false_pos), np.sum(true_neg)]],
                                index=['Y = 0', 'Y = 1'],
                                columns=['Yhat = 1', 'Yhat = 0'])
            # print(conf_mat)

            # print(confusion_matrix(y, logit_reg.predict(X)))
            print("\n*******************************")
            classificationSummary(y, logit_reg.predict(X), 
                                class_names=logit_reg.classes_)
            print("\n*******************************")
            
            self.prec_sens_espe(predictors,outcome,logit_reg)
            self.ROC_curva(predictors,outcome,logit_reg)
                        
        else:
            print(" El outcome no está bien escrito prueba con alguno de estos:\n")
            print(list(self.dummy.columns))
    
    def prec_sens_espe(self,predictors,outcome,logit_reg):
            y= self[outcome]
            X= self[predictors]
            conf_mat = confusion_matrix(y, logit_reg.predict(X))
            print(' - Precision', conf_mat[0, 0] / sum(conf_mat[:, 0]))
            print(' - Sensibilidad', conf_mat[0, 0] / sum(conf_mat[0, :]))
            print(' - Especificidad', conf_mat[1, 1] / sum(conf_mat[1, :]))

            precision_recall_fscore_support(y, logit_reg.predict(X), 
                                            labels=['0', '1'])
            

    def ROC_curva(self,predictors,outcome,logit_reg):

        y= self[outcome]
        X= self[predictors]
        fpr, tpr, thresholds = roc_curve(y, logit_reg.predict_proba(X)[:, 0], pos_label=0)
        roc_df = pd.DataFrame({'recall': tpr, 'specificity': 1 - fpr})
        ax = roc_df.plot(x='specificity', y='recall', figsize=(4, 4), legend=False)
        ax.set_ylim(0, 1)
        ax.set_xlim(1, 0)
        ax.plot((1, 0), (0, 1))
        ax.set_xlabel('Especificidad')
        ax.set_ylabel('Sensibilidad')
        plt.tight_layout()
        plt.show()

        
        fpr, tpr, thresholds = roc_curve(y, logit_reg.predict_proba(X)[:,0], 
                                        pos_label=0)
        roc_df = pd.DataFrame({'recall': tpr, 'specificity': 1 - fpr})

        ax = roc_df.plot(x='specificity', y='recall', figsize=(4, 4), legend=False)
        ax.set_ylim(0, 1)
        ax.set_xlim(1, 0)
        # ax.plot((1, 0), (0, 1))
        ax.set_xlabel('Especificidad')
        ax.set_ylabel('Sensibilidad')
        ax.fill_between(roc_df.specificity, 0, roc_df.recall, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def oversampling(self,predictors,outcome):
        X= self[predictors]
        y=self[outcome]
        positive_wt = 1 / np.mean(y==1)
        # default_wt = 1 / (np.sum(dummys.ES_NO_ES_s)/len(dummys.ES_NO_ES_s))
        # default_wt = 1 / np.mean(dummys.ES_NO_ES_s)
        wt = [positive_wt if outcome == 1 else 1 for outcome in y]

        full_model = LogisticRegression(penalty="l2", C=1e42, solver='liblinear')
        full_model.fit(X, y,wt)
        print('Porcentaje de valores predichos (weighting): ') 
        print( 100 * np.mean(full_model.predict(X) == 1) )  

    def data_gen(self,predictors,outcome):
        X= self[predictors]
        y=self[outcome]
        X_resampled, y_resampled = SMOTE().fit_resample(X, y)
        print('Porcentaje de elemetos positivos predichos (SMOTE resampled): ', 
            100 * np.mean(y_resampled == 1))
        
        full_model = LogisticRegression(penalty="l2", C=1e42, solver='liblinear')
        full_model.fit(X_resampled, y_resampled)
        print('Porcentaje de elemetos positivos predichos  (SMOTE): ', 
            100 * np.mean(full_model.predict(X) ==1 ))

    def explo_predict(self,predictors,outcome):
        if (outcome in list(self.dummy.columns)):
            y=self[outcome]
            X=self[predictors]
            loan_tree = DecisionTreeClassifier(random_state=1, criterion='entropy', 
                                            min_impurity_decrease=0.003)
            loan_tree.fit(X, y)

            loan_lda = LinearDiscriminantAnalysis()
            loan_lda.fit(X, y)

            logit_reg = LogisticRegression(penalty="l2", solver='liblinear')
            logit_reg.fit(X, y)


            ## model
            gam = LinearGAM(s(0) + s(1))
            print(gam.gridsearch(X.values, y.values))

            models = {
                'Decision Tree': loan_tree,
                'Linear Discriminant Analysis': loan_lda,
                'Logistic Regression': logit_reg,
                'Generalized Additive Model': gam,
            }

            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(5, 5))

            xvalues = np.arange(0.25, 0.73, 0.005)
            yvalues = np.arange(-0.1, 20.1, 0.1)
            xx, yy = np.meshgrid(xvalues, yvalues)
            X = pd.DataFrame({
                'Datos_F': xx.ravel(),
                'Datos_E': yy.ravel(),
            })

            boundary = {}

            for n, (title, model) in enumerate(models.items()):
                ax = axes[n // 2, n % 2]
                predict = model.predict(X)
                if 'Generalized' in title:
                    Z = np.array([1 if z > 0.5 else 0 for z in predict])
                else:
                    
                    Z = np.array([1 if z == 1 else 0 for z in predict])
                Z = Z.reshape(xx.shape)
                boundary[title] = yvalues[np.argmax(Z > 0, axis=0)]
                boundary[title][Z[-1,:] == 0] = yvalues[-1]

                c = ax.pcolormesh(xx, yy, Z, cmap='Blues', vmin=0.1, vmax=1.3, shading='auto')
                ax.set_title(title)
                ax.grid(True)

            plt.tight_layout()
            plt.show()



            boundary['Datos_F'] = xvalues
            boundaries = pd.DataFrame(boundary)

            fig, ax = plt.subplots(figsize=(5, 4))
            boundaries.plot(x='Datos_F', ax=ax)
            ax.set_ylabel('Datos_E')
            ax.set_ylim(0, 20)


            plt.tight_layout()
            plt.show()
        else:
            print(" El outcome no está bien escrito prueba con alguno de estos:\n")
            print(list(self.dummy.columns))

    def KNN_predict(self,predictors,outcome,new):
        y=self[outcome]
        X=self[predictors]
        knn = KNeighborsClassifier(n_neighbors=40)
        knn.fit(X, y)
        data=pd.DataFrame({'result':knn.predict_proba(new)[:,1]})
        print( data['result'])