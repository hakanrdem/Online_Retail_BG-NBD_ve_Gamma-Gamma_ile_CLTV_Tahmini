
# Online Retail BG-NBD ve Gamma-Gamma ile CLTV Prediction

# 1. Verinin Hazırlanması (Data Preperation)
# 2. BG-NBD Modeli ile Expected Number of Transaction
# 3. Gamma-Gamma Modeli ile Expected Average Profit
# 4. BG-NBD ve Gamma-Gamma Modeli ile CLTV'nin Hesaplanması
# 5. CLTV'ye Göre Segmentlerin Oluşturulması

# 1. Verinin Hazırlanması (Data Preperation)

# İngiltere merkezli perakende e-ticaret şirketi müşterilerini segmentlere ayırıp bu segmentlere göre
# pazarlama stratejileri belirlemek istiyor.

# Veri Seti Hikayesi

# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

# Online Retail II isimli veri seti İngiltere merkezli online bir satış mağazasının
# 01/12/2009 - 09/12/2011 tarihleri arasındaki satışlarını içeriyor.

# Değişkenler

# InvoiceNo: Fatura numarası. Her işleme yani faturaya ait eşsiz numara. C ile başlıyorsa iptal edilen işlem.
# StockCode: Ürün kodu. Her bir ürün için eşsiz numara.
# Description: Ürün ismi
# Quantity: Ürün adedi. Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.
# InvoiceDate: Fatura tarihi ve zamanı.
# UnitPrice: Ürün fiyatı (Sterlin cinsinden)
# CustomerID: Eşsiz müşteri numarası
# Country: Ülke ismi. Müşterinin yaşadığı ülke.

# Görev 1
# BG-NBD ve Gamma-Gamma Modellerini Kurarak 6 Aylık CLTV Tahmini Yapılması

# Verinin Hazırlanması (Data Preperation)

!pip install lifetimes
import datetime as dt
import pandas as pd
import matplotlib as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format" , lambda x: "%.4f" % x)
from sklearn.preprocessing import MinMaxScaler

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# Verininin Okunması

df_ = pd.read_excel("/Users/hakanerdem/PycharmProjects/pythonProject/dsmlbc_9_abdulkadir/Homeworks/hakan_erdem/2_CRM_Analitigi/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")
df = df_.copy()
df.describe().T
df.head()
df.isnull().sum()

# Verinin Temizlenmesi

df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

# Monetary değerinin hesaplanması için "TotalPrice" değişkeni
df["TotalPrice"] = df["Quantity"] * df["Price"]

today_date = dt.datetime(2011, 12, 11)

# CLTV Yapısının OLuşturulması

# recency: Son satın alma üzerinden geçen zaman. Haftalık. (kullanıcı özelinde)
# T: Müşterinin yaşı. Haftalık. (analiz tarihinden ne kadar süre önce ilk satın alma yapılmış)
# frequency: tekrar eden toplam satın alma sayısı (frequency>1)
# monetary: satın alma başına ortalama kazanç

cltv_df = df.groupby('Customer ID').agg(
                                         {'InvoiceDate': [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,
                                                          lambda InvoiceDate: (today_date - InvoiceDate.min()).days],
                                          'Invoice': lambda Invoice: Invoice.nunique(),
                                          'TotalPrice': lambda TotalPrice: TotalPrice.sum()})


cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
# satın alma başına ortalama kazanç
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
cltv_df.describe().T
# frequency: tekrar eden toplam satın alma sayısı (frequency>1)
cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7

# BG-NBD Modeli ile Expected Number of Transaction

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

# 6 Ayda Tüm Şirketin Beklenen Satış Sayısı Nedir?

bgf.predict(4 * 6,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sum()

cltv_df["expected_purc_6_month"] = bgf.predict(4 * 6,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])

#Gamma-Gamma Modeli ile Expected Average Profit

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary'])

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary'])

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])
cltv_df.sort_values("expected_average_profit", ascending=False)

cltv_df.head()

# BG-NBD ve Gamma-Gamma Modeli ile CLTV'nin Hesaplanması

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,  # 6 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv.head()

cltv = cltv.reset_index()
cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
cltv_final.sort_values(by="clv", ascending=False).head(10)

# Adım 2
# Elde ettiğiniz sonuçları yorumlayıp, değerlendiriniz.

####

# Görev 2
# Farklı Zaman Periyotlarından Oluşan CLTV Analizi

# Adım 1
# 2010-2011 UK müşterileri için 1 aylık ve 12 aylık CLTV hesaplayınız.

# BG-NBD Modeli ile Expected Number of Transaction
# 1 aylık CLTV hesaplayınız.

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

# 1 Ayda Tüm Şirketin Beklenen Satış Sayısı Nedir?

bgf.predict(4 * 1,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sum()

cltv_df["expected_purc_1_month"] = bgf.predict(4 * 1,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])

#Gamma-Gamma Modeli ile Expected Average Profit

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary'])

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary'])

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])
cltv_df.sort_values("expected_average_profit", ascending=False)

cltv_df.head()

# BG-NBD ve Gamma-Gamma Modeli ile CLTV'nin Hesaplanması

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=1,  # 1 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv.head()

cltv = cltv.reset_index()

# Adım 2
# 1 aylık CLTV'de en yüksek olan 10 kişi ile 12 aylık'taki en yüksek 10 kişiyi analiz ediniz.

cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
cltv_final.sort_values(by="clv", ascending=False).head(10)

"""
Customer ID
12347.0000    54.5224
12348.0000    45.3346
12352.0000    22.1299
12356.0000   120.5634
12358.0000   160.7478
Name: clv, dtype: float64

"""

# 12 aylık CLTV hesaplayınız.

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

# 1 Ayda Tüm Şirketin Beklenen Satış Sayısı Nedir?

bgf.predict(4 * 12,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sum()

cltv_df["expected_purc_12_month"] = bgf.predict(4 * 12,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])

#Gamma-Gamma Modeli ile Expected Average Profit

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary'])

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary'])

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])
cltv_df.sort_values("expected_average_profit", ascending=False)

cltv_df.head()

# BG-NBD ve Gamma-Gamma Modeli ile CLTV'nin Hesaplanması

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=12,  # 12 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv.head()

cltv = cltv.reset_index()

# Adım 2
# 1 aylık CLTV'de en yüksek olan 10 kişi ile 12 aylık'taki en yüksek 10 kişiyi analiz ediniz.

cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
cltv_final.sort_values(by="clv", ascending=False).head(10)

"""
Customer ID
12347.0000    598.0930
12348.0000    496.6408
12352.0000    241.5564
12356.0000   1317.0998
12358.0000   1719.3832
Name: clv, dtype: float64
"""

# Adım 3
# Fark var mı? Varsa sizce neden olabilir?

# Fark vardır çünkü bir müşterinin 12 aylık parasal katkısı 1 aylık katkıdan daha fazla olabilir.

"""
      Customer ID  recency       T  frequency  monetary  expected_purc_6_month  expected_average_profit  expected_purc_1_month  expected_purc_12_month        clv
1754   16000.0000   0.0000  0.4286          3  778.3733                 1.6639                 811.5693                 1.6639                 17.9368 14769.2038
2280   17084.0000   0.0000  5.1429          2  737.4375                 0.8578                 785.7015                 0.8578                  9.4018  7497.4340
49     12435.0000  26.8571 38.2857          2 1957.4725                 0.3041                2082.4309                 0.3041                  3.4935  7394.6897
1873   16240.0000   7.5714 11.1429          2  875.7300                 0.6858                 932.6874                 0.6858                  7.6320  7227.5728
31     12409.0000  14.7143 26.1429          3 1230.2967                 0.4674                1282.0432                 0.4674                  5.3210  6931.5505
843    14096.0000  13.8571 14.5714         17  186.0934                 2.8955                 187.6099                 2.8955                 32.5718  6206.6876
1534   15531.0000   3.1429  4.4286          2  501.8600                 0.9922                 535.3150                 0.9922                 10.8502  5894.7829
223    12762.0000   2.4286  3.7143          2  474.6300                 1.0375                 506.3732                 1.0375                 11.3185  5816.4190
2225   16984.0000   5.8571 18.7143          2 1120.3375                 0.4084                1192.6715                 0.4084                  4.6048  5578.5298
2774   18139.0000   0.0000  2.7143          6  234.3983                 2.0904                 239.7224                 2.0904                 22.7939  5545.2699

"""

# Görev 3
# Segmentasyon ve Aksiyon Önerileri

# Adım 1
# 2010-2011 UK müşterileri için 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.

# CLTV'ye Göre Segmentlerin Oluşturulması

cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

cltv_final.sort_values(by="clv", ascending=False).head(10)

cltv_final.groupby("segment").agg(
    {"count", "mean", "sum"})


"""
      Customer ID  recency       T  frequency  monetary  expected_purc_6_month  expected_average_profit  expected_purc_1_month  expected_purc_12_month       clv segment
1754   16000.0000   0.0000  0.4286          3  778.3733                 9.3767                 811.5693                 1.6639                 17.9368 7952.4577       A
2280   17084.0000   0.0000  5.1429          2  737.4375                 4.8888                 785.7015                 0.8578                  9.4018 4016.2597       A
49     12435.0000  26.8571 38.2857          2 1957.4725                 1.7842                2082.4309                 0.3041                  3.4935 3891.5655       A
1873   16240.0000   7.5714 11.1429          2  875.7300                 3.9474                 932.6874                 0.6858                  7.6320 3851.5524       A
31     12409.0000  14.7143 26.1429          3 1230.2967                 2.7281                1282.0432                 0.4674                  5.3210 3661.8853       A
843    14096.0000  13.8571 14.5714         17  186.0934                16.7786                 187.6099                 2.8955                 32.5718 3294.3177       A
1534   15531.0000   3.1429  4.4286          2  501.8600                 5.6463                 535.3150                 0.9922                 10.8502 3160.0849       A
223    12762.0000   2.4286  3.7143          2  474.6300                 5.8947                 506.3732                 1.0375                 11.3185 3120.4859       A
2774   18139.0000   0.0000  2.7143          6  234.3983                11.8723                 239.7224                 2.0904                 22.7939 2975.2744       A
2225   16984.0000   5.8571 18.7143          2 1120.3375                 2.3701                1192.6715                 0.4084                  4.6048 2958.5300       A

"""

# Adım 2
# 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz.

A grubunun sağlayacağı parsal katkı ile geride kalan 3 grubun sağladığı katkı toplam aynıdır. B, C ve d grupları tek bir grup haline getirilebilir.

"""
          Customer ID                     recency                        T               frequency                 monetary                expected_purc_6_month              expected_average_profit                expected_purc_1_month              expected_purc_12_month                      clv               
                  sum count       mean        sum count    mean        sum count    mean       sum count   mean         sum count     mean                   sum count   mean                     sum count     mean                   sum count   mean                    sum count   mean         sum count     mean
segment                                                                                                                                                                                                                                                                                                               
D       11081373.0000   712 15563.7261 20594.0000   712 28.9242 31679.2857   712 44.4934      4946   712 6.9466  34422.6972   712  48.3465             2148.0243   712 3.0169              36752.0891   712  51.6181              364.8870   712 0.5125              4217.3280   712 5.9232  55947.8078   712  78.5784
C       10966369.0000   711 15423.8664 25397.2857   711 35.7205 30119.4286   711 42.3621      4836   711 6.8017  41545.9068   711  58.4331             2603.5931   711 3.6619              43762.8433   711  61.5511              442.9837   711 0.6230              5105.7040   711 7.1810 121539.1291   711 170.9411
B       10844941.0000   711 15253.0816 21329.2857   711 29.9990 26076.5714   711 36.6759      4249   711 5.9761  69465.5290   711  97.7012             2650.0699   711 3.7272              73225.1293   711 102.9889              452.5950   711 0.6366              5183.4910   711 7.2904 200886.3687   711 282.5406
A       10613528.0000   711 14927.6062 13520.2857   711 19.0159 17302.0000   711 24.3347      3008   711 4.2307 144478.9215   711 203.2052             2991.1951   711 4.2070             152561.5738   711 214.5732              516.4278   711 0.7263              5809.6791   711 8.1711 539767.9366   711 759.1673

"""

