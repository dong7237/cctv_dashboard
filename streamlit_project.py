# 필요한 라이브러리 임포트
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import folium
import streamlit as st
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import *
from sklearn.model_selection import train_test_split,KFold, cross_val_score

with open('new_model.pkl', 'rb') as file:
    model = pickle.load(file)
# 스트림릿 앱 타이틀 설정
st.title("CCTV 설치 최적화 대시보드")

with st.sidebar:
    st.header("전처리된 데이터를 업로드 해주세요.")
    uploaded_file1 = st.file_uploader("인프라 데이터 파일", type=["csv", "xlsx"])
    uploaded_file2 = st.file_uploader("CCTV 데이터 파일", type=["csv", "xlsx"])
    uploaded_file3 = st.file_uploader("가로등 데이터 파일", type=["csv", "xlsx"])
    uploaded_file4 = st.file_uploader("공영주차장 데이터 파일", type=["csv", "xlsx"])
    uploaded_file5 = st.file_uploader("불법주정차 단속 데이터 파일", type=["csv", "xlsx"])

    st.header("입력값을 기반으로 예측을 수행합니다.")
    cluster_count = st.number_input("클러스터 개수", min_value=1, value=100)
    cctv_count = st.number_input("클러스터 당 CCTV 설치 대수", min_value=1, value=1)

if uploaded_file1 is not None:
    # 사용자가 업로드한 파일 처리
    if uploaded_file1.name.endswith('.csv'):
        infra_df = pd.read_csv(uploaded_file1)
        infra_df['정보'] = infra_df['정보'].str.replace('서울시광진구','',regex=False)
        infra_df['정보'] = infra_df['정보'].str.replace('인허가정보.csv','',regex=False)
        infra_inf= infra_df['정보'].value_counts().index
        infra_df = infra_df.rename(columns={'좌표정보(X)':'경도','좌표정보(Y)':'위도'})
        
        cctv = pd.read_csv(uploaded_file2)
        light = pd.read_csv(uploaded_file3)
        park = pd.read_csv(uploaded_file4)
        illigal = pd.read_csv(uploaded_file5)
        
        cctv=uploaded_file2
        cctv['정보'] = 'cctv'
        cctv = cctv[['위도','경도','정보']]
        park = uploaded_file3
        park['정보']='공영주차장'
        park = park[['위도','경도','정보']]
        light = uploaded_file4
        light['정보']= '가로등'
        light = light[['위도','경도','정보']]
        illigal = uploaded_file5
        illigal['정보'] = '단속'
        illigal = illigal[['위도','경도','정보']]
    else :
        st.error('csv파일로 업로드하세요')


    scores = 1.2
    #mae 모델 오차
    if st.button("예측 실행"):
        cluster_value = cluster_count
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=cluster_value, random_state=42)
        infra_df = infra_df.iloc[infra_df[['위도','경도']].dropna().index]
        infra_df['cluster'] = kmeans.fit_predict(infra_df[['위도','경도']].dropna())
        def clustering(df):
            df= df.iloc[df[['위도','경도']].dropna().index]
            df['cluster']=kmeans.predict(df[['위도','경도']].dropna())
            return df
        cctv = clustering(cctv)
        light = clustering(light)
        park = clustering(park)
        illigal = clustering(illigal)
        
        pivot=pd.pivot_table(data=infra_df,index='cluster',columns='정보',aggfunc='count',fill_value=0)
        pivot = pivot['경도']
        for i in [cctv,light,park,illigal]:
            t=pd.pivot_table(data=i,index='cluster',columns='정보',aggfunc='count',fill_value=0)['경도']
            pivot = pd.concat([pivot,t],axis=1)
            pivot = pivot.fillna(0)    
        
        data = pivot['단속']
        q1,q3 = np.percentile(data,[25,75])
        iqr = q3-q1
        lower_bound = q1-(iqr*1.5)
        upper_bound = q3+(iqr*1.5)
        pivot = pivot[(pivot['단속']>=lower_bound)&(pivot['단속']<=upper_bound)]
        
        test = pivot.copy()
        modify_cctv_values = cctv_count
        # 클러스터 당 변화 시킬 CCTV 대수
        test_pred = test.drop('단속',axis=1)
        pred_0 = model.predict(test_pred)
        test_pred['cctv'] = test_pred['cctv']+modify_cctv_values
        pred_= model.predict(test_pred) + np.mean(scores)
        result = pd.DataFrame([pred_,pred_0,test['단속']]).T
        result = result.dropna()
        result.columns = ['변화CCTV_예측','원본CCTV_예측','원본']
        result['예측값차이_절댓값'] = np.abs(result['원본']-result['원본CCTV_예측'])
        result['예측값간차이'] = result['변화CCTV_예측']-result['원본CCTV_예측']
        result['예측값간차이비'] = (result['변화CCTV_예측']/result['원본CCTV_예측'])*100
        result = result.sort_values(by='예측값차이_절댓값')
        result = result[result['원본']*0.2>=(result['예측값차이_절댓값'])]
        result = result.sort_values(by='예측값간차이비',ascending=False)
        max_eff_cluster = result.index[0]
        effect_ = result['예측값간차이비'][max_eff_cluster]
        
        tab1, tab2, tab3= st.tabs(['예측 결과' , '클러스터 위치 지도' , '기타 정보'])

        with tab1:
            st.header("예측 결과")
            st.write(f"설치 시 가장 큰 효과를 볼 수 있는 클러스터 값 : {max_eff_cluster}")
            st.write(f'설치 효과 : ',effect_,'%')
            st.write(f"클러스터 당 CCTV 추가설치 대수 : {modify_cctv_values}")

        with tab2:
            st.header("클러스터 위치")
            st.write("이 근방에 CCTV를 설치하는 것을 추천합니다.")
            max_effect_infra_df = infra_df[infra_df['cluster']==max_eff_cluster]

            seoul_map = folium.Map(location=[37.547790, 127.106990], zoom_start=13)
            for idx, row in max_effect_infra_df.iterrows():
                folium.Marker([row['위도'], row['경도']], popup=row['정보']).add_to(seoul_map)
            folium_static(seoul_map)
            
        with tab3:
            st.header("기타 정보 제공")
            feature_importance = pd.DataFrame(data=model.feature_importances_,index=pivot.columns.drop('단속'))*100
            feature_importance = feature_importance.sort_values(by=0,ascending=False)
            over_5_feature_importance = feature_importance[feature_importance[0]>=5]
            st.write(f"모델에 5% 이상 영향을 미친 인프라 정보 : ,{[i for i in over_5_feature_importance.index]}")
            bins = [-float('inf'),q1,q3,q3*1.5,float('inf')]
            labels = ['under_75%','over_low','over_mid','over_high']

            agu_df = pivot.copy()
            for i in range(100):
                t = pivot.sample(30)
                agu_df = pd.concat([agu_df,t])
            agu_df_insight = agu_df.copy()
            agu_df_insight['단속수준'] = pd.cut(agu_df['단속'],bins=bins,labels=labels)
            st.write("불법 주정차 단속수준 분류 데이터프레임")
            st.dataframe(agu_df_insight.groupby(by='단속수준').mean())
