import numpy as np 
import pandas as pd  
import os 
import seaborn as sns 
import matplotlib.pyplot as plt 
# import pandas_profiling as profile 
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
from sklearn.model_selection import train_test_split 
import pickle 
from scipy import stats 
from pylab import rcParams
import tensorflow as tf 

LABELS = ["Normal", "Fraud"]

#Loading Data sets

Train =pd.read_csv("D:\Msc\SEM5\DATA\Insurance data\kernel\Train.csv")
Train_Ben_data=pd.read_csv("D:\Msc\SEM5\DATA\Insurance data\kernel\Train_Beneficiarydata.csv")
Train_IP_data=pd.read_csv("D:\Msc\SEM5\DATA\Insurance data\kernel\Train_Inpatientdata.csv")
Train_OP_data=pd.read_csv("D:\Msc\SEM5\DATA\Insurance data\kernel\Train_Outpatientdata.csv")
Test =pd.read_csv("D:\Msc\SEM5\DATA\Insurance data\kernel\Test.csv")
Test_Ben_data =pd.read_csv("D:\Msc\SEM5\DATA\Insurance data\kernel\Test_Beneficiarydata.csv")
Test_IP_data =pd.read_csv("D:\Msc\SEM5\DATA\Insurance data\kernel\Test_Inpatientdata.csv")
Test_OP_data =pd.read_csv("D:\Msc\SEM5\DATA\Insurance data\kernel\Test_Outpatientdata.csv")
# print(Train_OP_data.shape)
# print(Train.Provider.value_counts(sort=True, ascending=False).head(2))

# print(outpatient.columns, outpatient.head() )
#checking missing values in datasets
#  print(Train_OP_data.isna().sum())

Train_Ben_data=Train_Ben_data.replace({'ChronicCond_Alzheimer': 2, 'ChronicCond_Heartfailure': 2, 'ChronicCond_KidneyDisease': 2,
'ChronicCond_Cancer': 2, 'ChronicCond_ObstrPulmonary': 2, 'ChronicCond_Depression': 2, 
'ChronicCond_Diabetes': 2, 'ChronicCond_IschemicHeart': 2, 'ChronicCond_Osteoporasis': 2, 
'ChronicCond_rheumatoidarthritis': 2, 'ChronicCond_stroke': 2 },0)
Train_Ben_data=Train_Ben_data.replace({'RenalDiseaseIndicator': 'Y'}, 1)
Test_Ben_data=Test_Ben_data.replace({'ChronicCond_Alzheimer': 2, 'ChronicCond_Heartfailure': 2, 'ChronicCond_KidneyDisease': 2,
'ChronicCond_Cancer': 2, 'ChronicCond_ObstrPulmonary': 2, 'ChronicCond_Depression': 2, 
'ChronicCond_Diabetes': 2, 'ChronicCond_IschemicHeart': 2, 'ChronicCond_Osteoporasis': 2, 
'ChronicCond_rheumatoidarthritis': 2, 'ChronicCond_stroke': 2 }, 0)
Test_Ben_data=Test_Ben_data.replace({'RenalDiseaseIndicator': 'Y'}, 1)
# print(Train_Ben_data.head())

# creating age column
Train_Ben_data['DOB'] = pd.to_datetime(Train_Ben_data['DOB'],format = '%Y-%m-%d')
Train_Ben_data['DOD'] = pd.to_datetime(Train_Ben_data['DOD'],format = '%Y-%m-%d',errors='ignore')
Train_Ben_data['Age'] = round(((Train_Ben_data['DOD'] - Train_Ben_data['DOB']).dt.days)/365)
Test_Ben_data['DOB'] = pd.to_datetime(Test_Ben_data['DOB'] , format = '%Y-%m-%d')
Test_Ben_data['DOD'] = pd.to_datetime(Test_Ben_data['DOD'],format = '%Y-%m-%d',errors='ignore')
Test_Ben_data['Age'] = round(((Test_Ben_data['DOD'] - Test_Ben_data['DOB']).dt.days)/365)

Train_Ben_data.Age.fillna(round(((pd.to_datetime('2009-12-01' , format = '%Y-%m-%d') - Train_Ben_data['DOB']).dt.days)/365),inplace=True)
Test_Ben_data.Age.fillna(round(((pd.to_datetime('2009-12-01' , format = '%Y-%m-%d') - Test_Ben_data['DOB']).dt.days)/365),inplace=True)

Train_Ben_data.loc[Train_Ben_data.DOD.isna(),'WhetherDead']=0
Train_Ben_data.loc[Train_Ben_data.DOD.notna(),'WhetherDead']=1
Train_Ben_data.loc[:,'WhetherDead'].head(7)


Test_Ben_data.loc[Test_Ben_data.DOD.isna(),'WhetherDead']=0
Test_Ben_data.loc[Test_Ben_data.DOD.notna(),'WhetherDead']=1
Test_Ben_data.loc[:,'WhetherDead'].head(3)
# print(Train_Ben_data.head(2))

## Merge Train_OP_data to Train_Beneficiarydata
Train_PatientDetaildata=pd.merge(Train_OP_data,Train_Ben_data,left_on='BeneID', right_on='BeneID', how='inner' )
Test_PatientDetaildata=pd.merge(Test_OP_data,Test_Ben_data, left_on='BeneID', right_on='BeneID', how='inner')

# print(Train_PatientDetaildata.shape)

## Merge Train_PatientDetaildata with fraudelent providers using "Provider" as joining key
Train_PatientDetail_labeldata=pd.merge(Train, Train_PatientDetaildata, on='Provider')
Test_PatientDetail_labeldata=pd.merge(Test, Test_PatientDetaildata, on='Provider')
# print(Train_PatientDetail_labeldata.shape)

####DESCRIPTIVE STATISTICS
describe=Train_PatientDetail_labeldata.describe()      #overall descriptive stats
class_counts=Train_PatientDetail_labeldata.groupby('PotentialFraud').size()   #class distribution
correlations=Train_PatientDetail_labeldata.corr(method='pearson')             #correlations btn attributes 
skew=Train_PatientDetail_labeldata.skew()                           #skew of univariate distribution
# print(skew)

sns.set_style('white',rc={'figure.figsize':(12,8)})
count_classes = pd.value_counts(Train_PatientDetail_labeldata['PotentialFraud'], sort = True)
print("Percent Distribution of Potential Fraud class:- \n",count_classes*100/len(Train_PatientDetail_labeldata))
LABELS = ["Non Fraud", "Fraud"]
#Drawing a barplot
count_classes.plot(kind = 'bar', rot=0,figsize=(10,6))

#Giving titles and labels to the plot
plt.title("Potential Fraud distribution in Aggregated claim transactional data")
plt.xticks(range(2), LABELS)
plt.xlabel("Potential Fraud Class ")
plt.ylabel("Number of PotentialFraud per Class ")

plt.savefig('PotentialFraudDistributionInMergedData')

#PLotting the frequencies of Statewise beneficiaries
count_States = pd.value_counts(Train_PatientDetail_labeldata['State'], sort = True)
#print("Percent Distribution of Beneficieries per state:- \n",count_States*100/len(Train_Beneficiarydata))

#Drawing a barplot
(count_States*100/len(Train_PatientDetail_labeldata)).plot(kind = 'bar', rot=0,figsize=(16,8),fontsize=12,legend=True)

#Giving titles and labels to the plot

# plt.annotate('Maximum Beneficiaries are from this State', xy=(0.01,8), xytext=(8, 6.5),
#              arrowprops=dict(facecolor='black', shrink=0.05))

plt.yticks(np.arange(0,10,2), ('0 %','2 %', '4 %', '6 %', '8 %', '10%'))
plt.title("State - wise Beneficiary Distribution",fontsize=18)
plt.xlabel("State Number",fontsize=15)
plt.ylabel("Percentage of Beneficiaries "'%',fontsize=15)
plt.show()

plt.savefig('StateWiseBeneficiaryDistribution')

#PLotting the frequencies of race-wise beneficiaries
count_Race = pd.value_counts(Train_PatientDetail_labeldata['Race'], sort = True)

#Drawing a barplot
(count_Race*100/len(Train_PatientDetail_labeldata)).plot(kind = 'bar', rot=0,figsize=(10,6),fontsize=12)

#Giving titles and labels to the plot
plt.yticks(np.arange(0,100,20))#, ('0 %','20 %', '40 %', '60 %', '80 %', '100%'))
plt.title("Race - wise Beneficiary Distribution",fontsize=18)
plt.xlabel("Race Code",fontsize=15)
plt.ylabel("Percentage of Beneficiaries "'%',fontsize=15)

plt.show()

plt.savefig('RacewiseBeneficiaryDistribution')

sns.set(rc={'figure.figsize':(12,8)},style='white')

ax=sns.countplot(x='State',hue='PotentialFraud',data=Train_PatientDetail_labeldata
              ,order=Train_PatientDetail_labeldata.Race.value_counts().iloc[:5].index)

plt.title('Top-5 State invloved in Healthcare Fraud')
    
plt.show()

plt.savefig('TopStateinvlovedinHealthcareFraud')

sns.set(rc={'figure.figsize':(12,8)},style='white')

ax=sns.countplot(x='Provider',hue='PotentialFraud', data=Train_PatientDetail_labeldata
              ,order=Train_PatientDetail_labeldata.Provider.value_counts().iloc[:10].index)

plt.title('Top-10 Provider invloved in Healthcare Fraud')
    
plt.show()

plt.savefig('TopProviderinvlovedinHealthcareFraud')

sns.set(rc={'figure.figsize':(12,8)},style='darkgrid')

ax=sns.countplot(x='OperatingPhysician',hue='PotentialFraud', data=Train_PatientDetail_labeldata
              ,order=Train_PatientDetail_labeldata.OperatingPhysician.value_counts().iloc[:10].index)

plt.title('Top-10 OperatingPhysician invloved in Healthcare Fraud')
    
plt.show()

plt.savefig('TopOperatingPhysicianrinvlovedinHealthcareFraud')

sns.set(rc={'figure.figsize':(10,8)},style='darkgrid')

ax=sns.countplot(x='ClmDiagnosisCode_1',hue='PotentialFraud', data=Train_PatientDetail_labeldata
              ,order=Train_PatientDetail_labeldata.ClmDiagnosisCode_1.value_counts().iloc[:10].index)

plt.title('Top-10 ClmDiagnosisCode_1 invloved in Healthcare Fraud')
    
plt.show()
plt.savefig('TopClmDiagnosisCode_1rinvlovedinHealthcareFraud')

sns.set(rc={'figure.figsize':(10,8)},style='darkgrid')

ax=sns.countplot(x='OPAnnualDeductibleAmt',hue='PotentialFraud', data=Train_PatientDetail_labeldata
              ,order=Train_PatientDetail_labeldata.OPAnnualDeductibleAmt.value_counts().iloc[:10].index)

plt.title('Top-10 OPAnnualDeductibleAmt invloved in Healthcare Fraud')
    
plt.show()

plt.savefig('TopOPAnnualDeductibleAmtinvlovedinHealthcareFraud')

sns.set(rc={'figure.figsize':(10,8)},style='darkgrid')

ax=sns.countplot(x='Race',hue='PotentialFraud', data=Train_PatientDetail_labeldata
              ,order=Train_PatientDetail_labeldata.Race.value_counts().iloc[:5].index)

plt.title('Top-10 Race invloved in Healthcare Fraud')
    
plt.show()

plt.savefig('TopRaceinvlovedinHealthcareFraud')

##AVERAGE FEATURES BASED ON GROUPING VARIABLES 
###Provider
Train_PatientDetail_labeldata["PerProviderAvg_DeductibleAmtPaid"]=Train_PatientDetail_labeldata.groupby('Provider')['DeductibleAmtPaid'].transform('mean')
Train_PatientDetail_labeldata["PerProviderAvg_IPAnnualReimbursementAmt"]=Train_PatientDetail_labeldata.groupby('Provider')['IPAnnualReimbursementAmt'].transform('mean')
Train_PatientDetail_labeldata["PerProviderAvg_IPAnnualDeductibleAmt"]=Train_PatientDetail_labeldata.groupby('Provider')['IPAnnualDeductibleAmt'].transform('mean')
Train_PatientDetail_labeldata["PerProviderAvg_OPAnnualReimbursementAmt"]=Train_PatientDetail_labeldata.groupby('Provider')['OPAnnualReimbursementAmt'].transform('mean')
Train_PatientDetail_labeldata["PerProviderAvg_OPAnnualDeductibleAmt"]=Train_PatientDetail_labeldata.groupby('Provider')['OPAnnualDeductibleAmt'].transform('mean')
Train_PatientDetail_labeldata["PerProviderAvg_Age"]=Train_PatientDetail_labeldata.groupby('Provider')['Age'].transform('mean')
Train_PatientDetail_labeldata["PerProviderAvg_NoOfMonths_PartACov"]=Train_PatientDetail_labeldata.groupby('Provider')['NoOfMonths_PartACov'].transform('mean')
Train_PatientDetail_labeldata["PerProviderAvg_NoOfMonths_PartBCov"]=Train_PatientDetail_labeldata.groupby('Provider')['NoOfMonths_PartBCov'].transform('mean')

Test_PatientDetail_labeldata["PerProviderAvg_DeductibleAmtPaid"]=Test_PatientDetail_labeldata.groupby('Provider')['DeductibleAmtPaid'].transform('mean')
Test_PatientDetail_labeldata["PerProviderAvg_IPAnnualReimbursementAmt"]=Test_PatientDetail_labeldata.groupby('Provider')['IPAnnualReimbursementAmt'].transform('mean')
Test_PatientDetail_labeldata["PerProviderAvg_IPAnnualDeductibleAmt"]=Test_PatientDetail_labeldata.groupby('Provider')['IPAnnualDeductibleAmt'].transform('mean')
Test_PatientDetail_labeldata["PerProviderAvg_OPAnnualReimbursementAmt"]=Test_PatientDetail_labeldata.groupby('Provider')['OPAnnualReimbursementAmt'].transform('mean')
Test_PatientDetail_labeldata["PerProviderAvg_OPAnnualDeductibleAmt"]=Test_PatientDetail_labeldata.groupby('Provider')['OPAnnualDeductibleAmt'].transform('mean')
Test_PatientDetail_labeldata["PerProviderAvg_Age"]=Test_PatientDetail_labeldata.groupby('Provider')['Age'].transform('mean')
Test_PatientDetail_labeldata["PerProviderAvg_NoOfMonths_PartACov"]=Test_PatientDetail_labeldata.groupby('Provider')['NoOfMonths_PartACov'].transform('mean')
Test_PatientDetail_labeldata["PerProviderAvg_NoOfMonths_PartBCov"]=Test_PatientDetail_labeldata.groupby('Provider')['NoOfMonths_PartBCov'].transform('mean')
# print(Train_PatientDetail_labeldata.shape)

##BeneID
Train_PatientDetail_labeldata["PerBeneIDAvg_DeductibleAmtPaid"]=Train_PatientDetail_labeldata.groupby('BeneID')['DeductibleAmtPaid'].transform('mean')
Train_PatientDetail_labeldata["PerBeneIDAvg_IPAnnualReimbursementAmt"]=Train_PatientDetail_labeldata.groupby('BeneID')['IPAnnualReimbursementAmt'].transform('mean')
Train_PatientDetail_labeldata["PerBeneIDAvg_IPAnnualDeductibleAmt"]=Train_PatientDetail_labeldata.groupby('BeneID')['IPAnnualDeductibleAmt'].transform('mean')
Train_PatientDetail_labeldata["PerBeneIDAvg_OPAnnualReimbursementAmt"]=Train_PatientDetail_labeldata.groupby('BeneID')['OPAnnualReimbursementAmt'].transform('mean')
Train_PatientDetail_labeldata["PerBeneIDAvg_OPAnnualDeductibleAmt"]=Train_PatientDetail_labeldata.groupby('BeneID')['OPAnnualDeductibleAmt'].transform('mean')

Test_PatientDetail_labeldata["PerBeneIDAvg_DeductibleAmtPaid"]=Test_PatientDetail_labeldata.groupby('BeneID')['DeductibleAmtPaid'].transform('mean')
Test_PatientDetail_labeldata["PerBeneIDAvg_IPAnnualReimbursementAmt"]=Test_PatientDetail_labeldata.groupby('BeneID')['IPAnnualReimbursementAmt'].transform('mean')
Test_PatientDetail_labeldata["PerBeneIDAvg_IPAnnualDeductibleAmt"]=Test_PatientDetail_labeldata.groupby('BeneID')['IPAnnualDeductibleAmt'].transform('mean')
Test_PatientDetail_labeldata["PerBeneIDAvg_OPAnnualReimbursementAmt"]=Test_PatientDetail_labeldata.groupby('BeneID')['OPAnnualReimbursementAmt'].transform('mean')
Test_PatientDetail_labeldata["PerBeneIDAvg_OPAnnualDeductibleAmt"]=Test_PatientDetail_labeldata.groupby('BeneID')['OPAnnualDeductibleAmt'].transform('mean')
# print(Test_PatientDetail_labeldata.shape)

###Operating Physician
Train_PatientDetail_labeldata["PerOperatingPhysicianAvg_DeductibleAmtPaid"]=Train_PatientDetail_labeldata.groupby('OperatingPhysician')['DeductibleAmtPaid'].transform('mean')
Train_PatientDetail_labeldata["PerOperatingPhysicianAvg_IPAnnualReimbursementAmt"]=Train_PatientDetail_labeldata.groupby('OperatingPhysician')['IPAnnualReimbursementAmt'].transform('mean')
Train_PatientDetail_labeldata["PerOperatingPhysicianAvg_IPAnnualDeductibleAmt"]=Train_PatientDetail_labeldata.groupby('OperatingPhysician')['IPAnnualDeductibleAmt'].transform('mean')
Train_PatientDetail_labeldata["PerOperatingPhysicianAvg_OPAnnualReimbursementAmt"]=Train_PatientDetail_labeldata.groupby('OperatingPhysician')['OPAnnualReimbursementAmt'].transform('mean')
Train_PatientDetail_labeldata["PerOperatingPhysicianAvg_OPAnnualDeductibleAmt"]=Train_PatientDetail_labeldata.groupby('OperatingPhysician')['OPAnnualDeductibleAmt'].transform('mean')

Test_PatientDetail_labeldata["PerOperatingPhysicianAvg_DeductibleAmtPaid"]=Test_PatientDetail_labeldata.groupby('OperatingPhysician')['DeductibleAmtPaid'].transform('mean')
Test_PatientDetail_labeldata["PerOperatingPhysicianAvg_IPAnnualReimbursementAmt"]=Test_PatientDetail_labeldata.groupby('OperatingPhysician')['IPAnnualReimbursementAmt'].transform('mean')
Test_PatientDetail_labeldata["PerOperatingPhysicianAvg_IPAnnualDeductibleAmt"]=Test_PatientDetail_labeldata.groupby('OperatingPhysician')['IPAnnualDeductibleAmt'].transform('mean')
Test_PatientDetail_labeldata["PerOperatingPhysicianAvg_OPAnnualReimbursementAmt"]=Test_PatientDetail_labeldata.groupby('OperatingPhysician')['OPAnnualReimbursementAmt'].transform('mean')
Test_PatientDetail_labeldata["PerOperatingPhysicianAvg_OPAnnualDeductibleAmt"]=Test_PatientDetail_labeldata.groupby('OperatingPhysician')['OPAnnualDeductibleAmt'].transform('mean')

###Attending Physician
Train_PatientDetail_labeldata["PerAttendingPhysicianAvg_DeductibleAmtPaid"]=Train_PatientDetail_labeldata.groupby('AttendingPhysician')['DeductibleAmtPaid'].transform('mean')
Train_PatientDetail_labeldata["PerAttendingPhysicianAvg_IPAnnualReimbursementAmt"]=Train_PatientDetail_labeldata.groupby('AttendingPhysician')['IPAnnualReimbursementAmt'].transform('mean')
Train_PatientDetail_labeldata["PerAttendingPhysicianAvg_IPAnnualDeductibleAmt"]=Train_PatientDetail_labeldata.groupby('AttendingPhysician')['IPAnnualDeductibleAmt'].transform('mean')
Train_PatientDetail_labeldata["PerAttendingPhysicianAvg_OPAnnualReimbursementAmt"]=Train_PatientDetail_labeldata.groupby('AttendingPhysician')['OPAnnualReimbursementAmt'].transform('mean')
Train_PatientDetail_labeldata["PerAttendingPhysicianAvg_OPAnnualDeductibleAmt"]=Train_PatientDetail_labeldata.groupby('AttendingPhysician')['OPAnnualDeductibleAmt'].transform('mean')

Test_PatientDetail_labeldata["PerAttendingPhysicianAvg_DeductibleAmtPaid"]=Test_PatientDetail_labeldata.groupby('AttendingPhysician')['DeductibleAmtPaid'].transform('mean')
Test_PatientDetail_labeldata["PerAttendingPhysicianAvg_IPAnnualReimbursementAmt"]=Test_PatientDetail_labeldata.groupby('AttendingPhysician')['IPAnnualReimbursementAmt'].transform('mean')
Test_PatientDetail_labeldata["PerAttendingPhysicianAvg_IPAnnualDeductibleAmt"]=Test_PatientDetail_labeldata.groupby('AttendingPhysician')['IPAnnualDeductibleAmt'].transform('mean')
Test_PatientDetail_labeldata["PerAttendingPhysicianAvg_OPAnnualReimbursementAmt"]=Test_PatientDetail_labeldata.groupby('AttendingPhysician')['OPAnnualReimbursementAmt'].transform('mean')
Test_PatientDetail_labeldata["PerAttendingPhysicianAvg_OPAnnualDeductibleAmt"]=Test_PatientDetail_labeldata.groupby('AttendingPhysician')['OPAnnualDeductibleAmt'].transform('mean')
# print(Train_PatientDetail_labeldata.columns)

###ClmAdmitDiagnosisCode
Train_PatientDetail_labeldata["PerClmAdmitDiagnosisCodeAvg_DeductibleAmtPaid"]=Train_PatientDetail_labeldata.groupby('ClmAdmitDiagnosisCode')['DeductibleAmtPaid'].transform('mean')
Train_PatientDetail_labeldata["PerClmAdmitDiagnosisCodeAvg_IPAnnualReimbursementAmt"]=Train_PatientDetail_labeldata.groupby('ClmAdmitDiagnosisCode')['IPAnnualReimbursementAmt'].transform('mean')
Train_PatientDetail_labeldata["PerClmAdmitDiagnosisCodeAvg_IPAnnualDeductibleAmt"]=Train_PatientDetail_labeldata.groupby('ClmAdmitDiagnosisCode')['IPAnnualDeductibleAmt'].transform('mean')
Train_PatientDetail_labeldata["PerClmAdmitDiagnosisCodeAvg_OPAnnualReimbursementAmt"]=Train_PatientDetail_labeldata.groupby('ClmAdmitDiagnosisCode')['OPAnnualReimbursementAmt'].transform('mean')
Train_PatientDetail_labeldata["PerClmAdmitDiagnosisCodeAvg_OPAnnualDeductibleAmt"]=Train_PatientDetail_labeldata.groupby('ClmAdmitDiagnosisCode')['OPAnnualDeductibleAmt'].transform('mean')

Test_PatientDetail_labeldata["PerClmAdmitDiagnosisCodeAvg_DeductibleAmtPaid"]=Test_PatientDetail_labeldata.groupby('ClmAdmitDiagnosisCode')['DeductibleAmtPaid'].transform('mean')
Test_PatientDetail_labeldata["PerClmAdmitDiagnosisCodeAvg_IPAnnualReimbursementAmt"]=Test_PatientDetail_labeldata.groupby('ClmAdmitDiagnosisCode')['IPAnnualReimbursementAmt'].transform('mean')
Test_PatientDetail_labeldata["PerClmAdmitDiagnosisCodeAvg_IPAnnualDeductibleAmt"]=Test_PatientDetail_labeldata.groupby('ClmAdmitDiagnosisCode')['IPAnnualDeductibleAmt'].transform('mean')
Test_PatientDetail_labeldata["PerClmAdmitDiagnosisCodeAvg_OPAnnualReimbursementAmt"]=Test_PatientDetail_labeldata.groupby('ClmAdmitDiagnosisCode')['OPAnnualReimbursementAmt'].transform('mean')
Test_PatientDetail_labeldata["PerClmAdmitDiagnosisCodeAvg_OPAnnualDeductibleAmt"]=Test_PatientDetail_labeldata.groupby('ClmAdmitDiagnosisCode')['OPAnnualDeductibleAmt'].transform('mean')
# print(Train_PatientDetail_labeldata.shape)
# ###ClmProcedureCode_1
Train_PatientDetail_labeldata["PerClmProcedureCode_1Avg_DeductibleAmtPaid"]=Train_PatientDetail_labeldata.groupby('ClmProcedureCode_1')['DeductibleAmtPaid'].transform('mean')
Train_PatientDetail_labeldata["PerClmProcedureCode_1Avg_IPAnnualReimbursementAmt"]=Train_PatientDetail_labeldata.groupby('ClmProcedureCode_1')['IPAnnualReimbursementAmt'].transform('mean')
Train_PatientDetail_labeldata["PerClmProcedureCode_1Avg_IPAnnualDeductibleAmt"]=Train_PatientDetail_labeldata.groupby('ClmProcedureCode_1')['IPAnnualDeductibleAmt'].transform('mean')
Train_PatientDetail_labeldata["PerClmProcedureCode_1Avg_OPAnnualReimbursementAmt"]=Train_PatientDetail_labeldata.groupby('ClmProcedureCode_1')['OPAnnualReimbursementAmt'].transform('mean')
Train_PatientDetail_labeldata["PerClmProcedureCode_1Avg_OPAnnualDeductibleAmt"]=Train_PatientDetail_labeldata.groupby('ClmProcedureCode_1')['OPAnnualDeductibleAmt'].transform('mean')

Test_PatientDetail_labeldata["PerClmProcedureCode_1Avg_DeductibleAmtPaid"]=Test_PatientDetail_labeldata.groupby('ClmProcedureCode_1')['DeductibleAmtPaid'].transform('mean')
Test_PatientDetail_labeldata["PerClmProcedureCode_1Avg_IPAnnualReimbursementAmt"]=Test_PatientDetail_labeldata.groupby('ClmProcedureCode_1')['IPAnnualReimbursementAmt'].transform('mean')
Test_PatientDetail_labeldata["PerClmProcedureCode_1Avg_IPAnnualDeductibleAmt"]=Test_PatientDetail_labeldata.groupby('ClmProcedureCode_1')['IPAnnualDeductibleAmt'].transform('mean')
Test_PatientDetail_labeldata["PerClmProcedureCode_1Avg_OPAnnualReimbursementAmt"]=Test_PatientDetail_labeldata.groupby('ClmProcedureCode_1')['OPAnnualReimbursementAmt'].transform('mean')
Test_PatientDetail_labeldata["PerClmProcedureCode_1Avg_OPAnnualDeductibleAmt"]=Test_PatientDetail_labeldata.groupby('ClmProcedureCode_1')['OPAnnualDeductibleAmt'].transform('mean')
# print(Test_PatientDetail_labeldata.shape)
# ###ClmProcedureCode_2
# ###ClmProcedureCode_3
Train_PatientDetail_labeldata["PerClmProcedureCode_3Avg_DeductibleAmtPaid"]=Train_PatientDetail_labeldata.groupby('ClmProcedureCode_3')['DeductibleAmtPaid'].transform('mean')
Train_PatientDetail_labeldata["PerClmProcedureCode_3Avg_IPAnnualReimbursementAmt"]=Train_PatientDetail_labeldata.groupby('ClmProcedureCode_3')['IPAnnualReimbursementAmt'].transform('mean')
Train_PatientDetail_labeldata["PerClmProcedureCode_3Avg_IPAnnualDeductibleAmt"]=Train_PatientDetail_labeldata.groupby('ClmProcedureCode_3')['IPAnnualDeductibleAmt'].transform('mean')
Train_PatientDetail_labeldata["PerClmProcedureCode_3Avg_OPAnnualReimbursementAmt"]=Train_PatientDetail_labeldata.groupby('ClmProcedureCode_3')['OPAnnualReimbursementAmt'].transform('mean')
Train_PatientDetail_labeldata["PerClmProcedureCode_3Avg_OPAnnualDeductibleAmt"]=Train_PatientDetail_labeldata.groupby('ClmProcedureCode_3')['OPAnnualDeductibleAmt'].transform('mean')

Test_PatientDetail_labeldata["PerClmProcedureCode_3Avg_DeductibleAmtPaid"]=Test_PatientDetail_labeldata.groupby('ClmProcedureCode_3')['DeductibleAmtPaid'].transform('mean')
Test_PatientDetail_labeldata["PerClmProcedureCode_3Avg_IPAnnualReimbursementAmt"]=Test_PatientDetail_labeldata.groupby('ClmProcedureCode_3')['IPAnnualReimbursementAmt'].transform('mean')
Test_PatientDetail_labeldata["PerClmProcedureCode_3Avg_IPAnnualDeductibleAmt"]=Test_PatientDetail_labeldata.groupby('ClmProcedureCode_3')['IPAnnualDeductibleAmt'].transform('mean')
Test_PatientDetail_labeldata["PerClmProcedureCode_3Avg_OPAnnualReimbursementAmt"]=Test_PatientDetail_labeldata.groupby('ClmProcedureCode_3')['OPAnnualReimbursementAmt'].transform('mean')
Test_PatientDetail_labeldata["PerClmProcedureCode_3Avg_OPAnnualDeductibleAmt"]=Test_PatientDetail_labeldata.groupby('ClmProcedureCode_3')['OPAnnualDeductibleAmt'].transform('mean')
# print(Test_PatientDetail_labeldata.shape)
# ###ClmDiagnosisCode_1
Train_PatientDetail_labeldata["PerClmDiagnosisCode_1Avg_DeductibleAmtPaid"]=Train_PatientDetail_labeldata.groupby('ClmDiagnosisCode_1')['DeductibleAmtPaid'].transform('mean')
Train_PatientDetail_labeldata["PerClmDiagnosisCode_1Avg_IPAnnualReimbursementAmt"]=Train_PatientDetail_labeldata.groupby('ClmDiagnosisCode_1')['IPAnnualReimbursementAmt'].transform('mean')
Train_PatientDetail_labeldata["PerClmDiagnosisCode_1Avg_IPAnnualDeductibleAmt"]=Train_PatientDetail_labeldata.groupby('ClmDiagnosisCode_1')['IPAnnualDeductibleAmt'].transform('mean')
Train_PatientDetail_labeldata["PerClmDiagnosisCode_1Avg_OPAnnualReimbursementAmt"]=Train_PatientDetail_labeldata.groupby('ClmDiagnosisCode_1')['OPAnnualReimbursementAmt'].transform('mean')
Train_PatientDetail_labeldata["PerClmDiagnosisCode_1Avg_OPAnnualDeductibleAmt"]=Train_PatientDetail_labeldata.groupby('ClmDiagnosisCode_1')['OPAnnualDeductibleAmt'].transform('mean')

Test_PatientDetail_labeldata["PerClmDiagnosisCode_1Avg_DeductibleAmtPaid"]=Test_PatientDetail_labeldata.groupby('ClmDiagnosisCode_1')['DeductibleAmtPaid'].transform('mean')
Test_PatientDetail_labeldata["PerClmDiagnosisCode_1Avg_IPAnnualReimbursementAmt"]=Test_PatientDetail_labeldata.groupby('ClmDiagnosisCode_1')['IPAnnualReimbursementAmt'].transform('mean')
Test_PatientDetail_labeldata["PerClmDiagnosisCode_1Avg_IPAnnualDeductibleAmt"]=Test_PatientDetail_labeldata.groupby('ClmDiagnosisCode_1')['IPAnnualDeductibleAmt'].transform('mean')
Test_PatientDetail_labeldata["PerClmDiagnosisCode_1Avg_OPAnnualReimbursementAmt"]=Test_PatientDetail_labeldata.groupby('ClmDiagnosisCode_1')['OPAnnualReimbursementAmt'].transform('mean')
Test_PatientDetail_labeldata["PerClmDiagnosisCode_1Avg_OPAnnualDeductibleAmt"]=Test_PatientDetail_labeldata.groupby('ClmDiagnosisCode_1')['OPAnnualDeductibleAmt'].transform('mean')
# print(Train_PatientDetail_labeldata.shape)
# ###ClmDiagnosisCode_2
Train_PatientDetail_labeldata["PerClmDiagnosisCode_2Avg_DeductibleAmtPaid"]=Train_PatientDetail_labeldata.groupby('ClmDiagnosisCode_2')['DeductibleAmtPaid'].transform('mean')
Train_PatientDetail_labeldata["PerClmDiagnosisCode_2Avg_IPAnnualReimbursementAmt"]=Train_PatientDetail_labeldata.groupby('ClmDiagnosisCode_2')['IPAnnualReimbursementAmt'].transform('mean')
Train_PatientDetail_labeldata["PerClmDiagnosisCode_2Avg_IPAnnualDeductibleAmt"]=Train_PatientDetail_labeldata.groupby('ClmDiagnosisCode_2')['IPAnnualDeductibleAmt'].transform('mean')
Train_PatientDetail_labeldata["PerClmDiagnosisCode_2Avg_OPAnnualReimbursementAmt"]=Train_PatientDetail_labeldata.groupby('ClmDiagnosisCode_2')['OPAnnualReimbursementAmt'].transform('mean')
Train_PatientDetail_labeldata["PerClmDiagnosisCode_2Avg_OPAnnualDeductibleAmt"]=Train_PatientDetail_labeldata.groupby('ClmDiagnosisCode_2')['OPAnnualDeductibleAmt'].transform('mean')

Test_PatientDetail_labeldata["PerClmDiagnosisCode_2Avg_DeductibleAmtPaid"]=Test_PatientDetail_labeldata.groupby('ClmDiagnosisCode_2')['DeductibleAmtPaid'].transform('mean')
Test_PatientDetail_labeldata["PerClmDiagnosisCode_2Avg_IPAnnualReimbursementAmt"]=Test_PatientDetail_labeldata.groupby('ClmDiagnosisCode_2')['IPAnnualReimbursementAmt'].transform('mean')
Test_PatientDetail_labeldata["PerClmDiagnosisCode_2Avg_IPAnnualDeductibleAmt"]=Test_PatientDetail_labeldata.groupby('ClmDiagnosisCode_2')['IPAnnualDeductibleAmt'].transform('mean')
Test_PatientDetail_labeldata["PerClmDiagnosisCode_2Avg_OPAnnualReimbursementAmt"]=Test_PatientDetail_labeldata.groupby('ClmDiagnosisCode_2')['OPAnnualReimbursementAmt'].transform('mean')
Test_PatientDetail_labeldata["PerClmDiagnosisCode_2Avg_OPAnnualDeductibleAmt"]=Test_PatientDetail_labeldata.groupby('ClmDiagnosisCode_2')['OPAnnualDeductibleAmt'].transform('mean')

# ###ClmDiagnosisCode_3
Train_PatientDetail_labeldata["PerClmDiagnosisCode_3Avg_DeductibleAmtPaid"]=Train_PatientDetail_labeldata.groupby('ClmDiagnosisCode_3')['DeductibleAmtPaid'].transform('mean')
Train_PatientDetail_labeldata["PerClmDiagnosisCode_3Avg_IPAnnualReimbursementAmt"]=Train_PatientDetail_labeldata.groupby('ClmDiagnosisCode_3')['IPAnnualReimbursementAmt'].transform('mean')
Train_PatientDetail_labeldata["PerClmDiagnosisCode_3Avg_IPAnnualDeductibleAmt"]=Train_PatientDetail_labeldata.groupby('ClmDiagnosisCode_3')['IPAnnualDeductibleAmt'].transform('mean')
Train_PatientDetail_labeldata["PerClmDiagnosisCode_3Avg_OPAnnualReimbursementAmt"]=Train_PatientDetail_labeldata.groupby('ClmDiagnosisCode_3')['OPAnnualReimbursementAmt'].transform('mean')
Train_PatientDetail_labeldata["PerClmDiagnosisCode_3Avg_OPAnnualDeductibleAmt"]=Train_PatientDetail_labeldata.groupby('ClmDiagnosisCode_3')['OPAnnualDeductibleAmt'].transform('mean')

Test_PatientDetail_labeldata["PerClmDiagnosisCode_3Avg_DeductibleAmtPaid"]=Test_PatientDetail_labeldata.groupby('ClmDiagnosisCode_3')['DeductibleAmtPaid'].transform('mean')
Test_PatientDetail_labeldata["PerClmDiagnosisCode_3Avg_IPAnnualReimbursementAmt"]=Test_PatientDetail_labeldata.groupby('ClmDiagnosisCode_3')['IPAnnualReimbursementAmt'].transform('mean')
Test_PatientDetail_labeldata["PerClmDiagnosisCode_3Avg_IPAnnualDeductibleAmt"]=Test_PatientDetail_labeldata.groupby('ClmDiagnosisCode_3')['IPAnnualDeductibleAmt'].transform('mean')
Test_PatientDetail_labeldata["PerClmDiagnosisCode_3Avg_OPAnnualReimbursementAmt"]=Test_PatientDetail_labeldata.groupby('ClmDiagnosisCode_3')['OPAnnualReimbursementAmt'].transform('mean')
Test_PatientDetail_labeldata["PerClmDiagnosisCode_3Avg_OPAnnualDeductibleAmt"]=Test_PatientDetail_labeldata.groupby('ClmDiagnosisCode_3')['OPAnnualDeductibleAmt'].transform('mean')

# ###ClmDiagnosisCode_4
Train_PatientDetail_labeldata["PerClmDiagnosisCode_4Avg_DeductibleAmtPaid"]=Train_PatientDetail_labeldata.groupby('ClmDiagnosisCode_4')['DeductibleAmtPaid'].transform('mean')
Train_PatientDetail_labeldata["PerClmDiagnosisCode_4Avg_IPAnnualReimbursementAmt"]=Train_PatientDetail_labeldata.groupby('ClmDiagnosisCode_4')['IPAnnualReimbursementAmt'].transform('mean')
Train_PatientDetail_labeldata["PerClmDiagnosisCode_4Avg_IPAnnualDeductibleAmt"]=Train_PatientDetail_labeldata.groupby('ClmDiagnosisCode_4')['IPAnnualDeductibleAmt'].transform('mean')
Train_PatientDetail_labeldata["PerClmDiagnosisCode_4Avg_OPAnnualReimbursementAmt"]=Train_PatientDetail_labeldata.groupby('ClmDiagnosisCode_4')['OPAnnualReimbursementAmt'].transform('mean')
Train_PatientDetail_labeldata["PerClmDiagnosisCode_4Avg_OPAnnualDeductibleAmt"]=Train_PatientDetail_labeldata.groupby('ClmDiagnosisCode_4')['OPAnnualDeductibleAmt'].transform('mean')

Test_PatientDetail_labeldata["PerClmDiagnosisCode_4Avg_DeductibleAmtPaid"]=Test_PatientDetail_labeldata.groupby('ClmDiagnosisCode_4')['DeductibleAmtPaid'].transform('mean')
Test_PatientDetail_labeldata["PerClmDiagnosisCode_4Avg_IPAnnualReimbursementAmt"]=Test_PatientDetail_labeldata.groupby('ClmDiagnosisCode_4')['IPAnnualReimbursementAmt'].transform('mean')
Test_PatientDetail_labeldata["PerClmDiagnosisCode_4Avg_IPAnnualDeductibleAmt"]=Test_PatientDetail_labeldata.groupby('ClmDiagnosisCode_4')['IPAnnualDeductibleAmt'].transform('mean')
Test_PatientDetail_labeldata["PerClmDiagnosisCode_4Avg_OPAnnualReimbursementAmt"]=Test_PatientDetail_labeldata.groupby('ClmDiagnosisCode_4')['OPAnnualReimbursementAmt'].transform('mean')
Test_PatientDetail_labeldata["PerClmDiagnosisCode_4Avg_OPAnnualDeductibleAmt"]=Test_PatientDetail_labeldata.groupby('ClmDiagnosisCode_4')['OPAnnualDeductibleAmt'].transform('mean')
# print(Train_PatientDetail_labeldata.shape)
###Combinations of different variables 
# Train_PatientDetail_labeldata["ClmCount_Provider"]=Train_PatientDetail_labeldata.groupby(['Provider'])['ClaimID'].transform('count')
Train_PatientDetail_labeldata["ClmCount_Provider_BeneID"]=Train_PatientDetail_labeldata.groupby(['Provider','BeneID'])['ClaimID'].transform('count')
Train_PatientDetail_labeldata["ClmCount_Provider_AttendingPhysician"]=Train_PatientDetail_labeldata.groupby(['Provider','AttendingPhysician'])['ClaimID'].transform('count')
Train_PatientDetail_labeldata["ClmCount_Provider_OtherPhysician"]=Train_PatientDetail_labeldata.groupby(['Provider','OtherPhysician'])['ClaimID'].transform('count')
Train_PatientDetail_labeldata["ClmCount_Provider_OperatingPhysician"]=Train_PatientDetail_labeldata.groupby(['Provider','OperatingPhysician'])['ClaimID'].transform('count')
Train_PatientDetail_labeldata["ClmCount_Provider_ClmAdmitDiagnosisCode"]=Train_PatientDetail_labeldata.groupby(['Provider','ClmAdmitDiagnosisCode'])['ClaimID'].transform('count')
Train_PatientDetail_labeldata["ClmCount_Provider_ClmProcedureCode_1"]=Train_PatientDetail_labeldata.groupby(['Provider','ClmProcedureCode_1'])['ClaimID'].transform('count')
Train_PatientDetail_labeldata["ClmCount_Provider_ClmProcedureCode_2"]=Train_PatientDetail_labeldata.groupby(['Provider','ClmProcedureCode_2'])['ClaimID'].transform('count')
Train_PatientDetail_labeldata["ClmCount_Provider_ClmProcedureCode_3"]=Train_PatientDetail_labeldata.groupby(['Provider','ClmProcedureCode_3'])['ClaimID'].transform('count')
Train_PatientDetail_labeldata["ClmCount_Provider_ClmProcedureCode_4"]=Train_PatientDetail_labeldata.groupby(['Provider','ClmProcedureCode_4'])['ClaimID'].transform('count')
# Train_PatientDetail_labeldata["ClmCount_Provider_ClmProcedureCode_5"]=Train_PatientDetail_labeldata.groupby(['Provider','ClmProcedureCode_5'])['ClaimID'].transform('count')
Train_PatientDetail_labeldata["ClmCount_Provider_ClmDiagnosisCode_1"]=Train_PatientDetail_labeldata.groupby(['Provider','ClmDiagnosisCode_1'])['ClaimID'].transform('count')
Train_PatientDetail_labeldata["ClmCount_Provider_ClmDiagnosisCode_2"]=Train_PatientDetail_labeldata.groupby(['Provider','ClmDiagnosisCode_2'])['ClaimID'].transform('count')
Train_PatientDetail_labeldata["ClmCount_Provider_ClmDiagnosisCode_3"]=Train_PatientDetail_labeldata.groupby(['Provider','ClmDiagnosisCode_3'])['ClaimID'].transform('count')
Train_PatientDetail_labeldata["ClmCount_Provider_ClmDiagnosisCode_4"]=Train_PatientDetail_labeldata.groupby(['Provider','ClmDiagnosisCode_4'])['ClaimID'].transform('count')
Train_PatientDetail_labeldata["ClmCount_Provider_ClmDiagnosisCode_5"]=Train_PatientDetail_labeldata.groupby(['Provider','ClmDiagnosisCode_5'])['ClaimID'].transform('count')
Train_PatientDetail_labeldata["ClmCount_Provider_ClmDiagnosisCode_6"]=Train_PatientDetail_labeldata.groupby(['Provider','ClmDiagnosisCode_6'])['ClaimID'].transform('count')
Train_PatientDetail_labeldata["ClmCount_Provider_ClmDiagnosisCode_7"]=Train_PatientDetail_labeldata.groupby(['Provider','ClmDiagnosisCode_7'])['ClaimID'].transform('count')
Train_PatientDetail_labeldata["ClmCount_Provider_ClmDiagnosisCode_8"]=Train_PatientDetail_labeldata.groupby(['Provider','ClmDiagnosisCode_8'])['ClaimID'].transform('count')
Train_PatientDetail_labeldata["ClmCount_Provider_ClmDiagnosisCode_9"]=Train_PatientDetail_labeldata.groupby(['Provider','ClmDiagnosisCode_9'])['ClaimID'].transform('count')
Train_PatientDetail_labeldata["ClmCount_Provider_BeneID_AttendingPhysician"]=Train_PatientDetail_labeldata.groupby(['Provider','BeneID','AttendingPhysician'])['ClaimID'].transform('count')
Train_PatientDetail_labeldata["ClmCount_Provider_BeneID_OtherPhysician"]=Train_PatientDetail_labeldata.groupby(['Provider','BeneID','OtherPhysician'])['ClaimID'].transform('count')
Train_PatientDetail_labeldata["ClmCount_Provider_BeneID_AttendingPhysician_ClmProcedureCode_1"]=Train_PatientDetail_labeldata.groupby(['Provider','BeneID','AttendingPhysician','ClmProcedureCode_1'])['ClaimID'].transform('count')
Train_PatientDetail_labeldata["ClmCount_Provider_BeneID_AttendingPhysician_ClmDiagnosisCode_1"]=Train_PatientDetail_labeldata.groupby(['Provider','BeneID','AttendingPhysician','ClmDiagnosisCode_1'])['ClaimID'].transform('count')
Train_PatientDetail_labeldata["ClmCount_Provider_BeneID_OperatingPhysician"]=Train_PatientDetail_labeldata.groupby(['Provider','BeneID','OperatingPhysician'])['ClaimID'].transform('count')
Train_PatientDetail_labeldata["ClmCount_Provider_BeneID_ClmProcedureCode_1"]=Train_PatientDetail_labeldata.groupby(['Provider','BeneID','ClmProcedureCode_1'])['ClaimID'].transform('count')
Train_PatientDetail_labeldata["ClmCount_Provider_BeneID_ClmDiagnosisCode_1"]=Train_PatientDetail_labeldata.groupby(['Provider','BeneID','ClmDiagnosisCode_1'])['ClaimID'].transform('count')
Train_PatientDetail_labeldata["ClmCount_Provider_BeneID_ClmDiagnosisCode_1_ClmProcedureCode_1"]=Train_PatientDetail_labeldata.groupby(['Provider','BeneID','ClmDiagnosisCode_1','ClmProcedureCode_1'])['ClaimID'].transform('count')

# Test_PatientDetail_labeldata["ClmCount_Provider"]=Test_PatientDetail_labeldata.groupby(['Provider'])['ClaimID'].transform('count')
Test_PatientDetail_labeldata["ClmCount_Provider_BeneID"]=Test_PatientDetail_labeldata.groupby(['Provider','BeneID'])['ClaimID'].transform('count')
Test_PatientDetail_labeldata["ClmCount_Provider_AttendingPhysician"]=Test_PatientDetail_labeldata.groupby(['Provider','AttendingPhysician'])['ClaimID'].transform('count')
Test_PatientDetail_labeldata["ClmCount_Provider_OtherPhysician"]=Test_PatientDetail_labeldata.groupby(['Provider','OtherPhysician'])['ClaimID'].transform('count')
Test_PatientDetail_labeldata["ClmCount_Provider_OperatingPhysician"]=Test_PatientDetail_labeldata.groupby(['Provider','OperatingPhysician'])['ClaimID'].transform('count')
Test_PatientDetail_labeldata["ClmCount_Provider_ClmAdmitDiagnosisCode"]=Test_PatientDetail_labeldata.groupby(['Provider','ClmAdmitDiagnosisCode'])['ClaimID'].transform('count')
Test_PatientDetail_labeldata["ClmCount_Provider_ClmProcedureCode_1"]=Test_PatientDetail_labeldata.groupby(['Provider','ClmProcedureCode_1'])['ClaimID'].transform('count')
Test_PatientDetail_labeldata["ClmCount_Provider_ClmProcedureCode_2"]=Test_PatientDetail_labeldata.groupby(['Provider','ClmProcedureCode_2'])['ClaimID'].transform('count')
Test_PatientDetail_labeldata["ClmCount_Provider_ClmProcedureCode_3"]=Test_PatientDetail_labeldata.groupby(['Provider','ClmProcedureCode_3'])['ClaimID'].transform('count')
Test_PatientDetail_labeldata["ClmCount_Provider_ClmProcedureCode_4"]=Test_PatientDetail_labeldata.groupby(['Provider','ClmProcedureCode_4'])['ClaimID'].transform('count')
# Test_PatientDetail_labeldata["ClmCount_Provider_ClmProcedureCode_5"]=Test_PatientDetail_labeldata.groupby(['Provider','ClmProcedureCode_5'])['ClaimID'].transform('count')
Test_PatientDetail_labeldata["ClmCount_Provider_ClmDiagnosisCode_1"]=Test_PatientDetail_labeldata.groupby(['Provider','ClmDiagnosisCode_1'])['ClaimID'].transform('count')
Test_PatientDetail_labeldata["ClmCount_Provider_ClmDiagnosisCode_2"]=Test_PatientDetail_labeldata.groupby(['Provider','ClmDiagnosisCode_2'])['ClaimID'].transform('count')
Test_PatientDetail_labeldata["ClmCount_Provider_ClmDiagnosisCode_3"]=Test_PatientDetail_labeldata.groupby(['Provider','ClmDiagnosisCode_3'])['ClaimID'].transform('count')
Test_PatientDetail_labeldata["ClmCount_Provider_ClmDiagnosisCode_4"]=Test_PatientDetail_labeldata.groupby(['Provider','ClmDiagnosisCode_4'])['ClaimID'].transform('count')
Test_PatientDetail_labeldata["ClmCount_Provider_ClmDiagnosisCode_5"]=Test_PatientDetail_labeldata.groupby(['Provider','ClmDiagnosisCode_5'])['ClaimID'].transform('count')
Test_PatientDetail_labeldata["ClmCount_Provider_ClmDiagnosisCode_6"]=Test_PatientDetail_labeldata.groupby(['Provider','ClmDiagnosisCode_6'])['ClaimID'].transform('count')
Test_PatientDetail_labeldata["ClmCount_Provider_ClmDiagnosisCode_7"]=Test_PatientDetail_labeldata.groupby(['Provider','ClmDiagnosisCode_7'])['ClaimID'].transform('count')
Test_PatientDetail_labeldata["ClmCount_Provider_ClmDiagnosisCode_8"]=Test_PatientDetail_labeldata.groupby(['Provider','ClmDiagnosisCode_8'])['ClaimID'].transform('count')
Test_PatientDetail_labeldata["ClmCount_Provider_ClmDiagnosisCode_9"]=Test_PatientDetail_labeldata.groupby(['Provider','ClmDiagnosisCode_9'])['ClaimID'].transform('count')
Test_PatientDetail_labeldata["ClmCount_Provider_BeneID_AttendingPhysician"]=Test_PatientDetail_labeldata.groupby(['Provider','BeneID','AttendingPhysician'])['ClaimID'].transform('count')
Test_PatientDetail_labeldata["ClmCount_Provider_BeneID_OtherPhysician"]=Test_PatientDetail_labeldata.groupby(['Provider','BeneID','OtherPhysician'])['ClaimID'].transform('count')
Test_PatientDetail_labeldata["ClmCount_Provider_BeneID_AttendingPhysician_ClmProcedureCode_1"]=Test_PatientDetail_labeldata.groupby(['Provider','BeneID','AttendingPhysician','ClmProcedureCode_1'])['ClaimID'].transform('count')
Test_PatientDetail_labeldata["ClmCount_Provider_BeneID_AttendingPhysician_ClmDiagnosisCode_1"]=Test_PatientDetail_labeldata.groupby(['Provider','BeneID','AttendingPhysician','ClmDiagnosisCode_1'])['ClaimID'].transform('count')
Test_PatientDetail_labeldata["ClmCount_Provider_BeneID_OperatingPhysician"]=Test_PatientDetail_labeldata.groupby(['Provider','BeneID','OperatingPhysician'])['ClaimID'].transform('count')
Test_PatientDetail_labeldata["ClmCount_Provider_BeneID_ClmProcedureCode_1"]=Test_PatientDetail_labeldata.groupby(['Provider','BeneID','ClmProcedureCode_1'])['ClaimID'].transform('count')
Test_PatientDetail_labeldata["ClmCount_Provider_BeneID_ClmDiagnosisCode_1"]=Test_PatientDetail_labeldata.groupby(['Provider','BeneID','ClmDiagnosisCode_1'])['ClaimID'].transform('count')
Test_PatientDetail_labeldata["ClmCount_Provider_BeneID_ClmDiagnosisCode_1_ClmProcedureCode_1"]=Test_PatientDetail_labeldata.groupby(['Provider','BeneID','ClmDiagnosisCode_1','ClmProcedureCode_1'])['ClaimID'].transform('count')
# print('Test_ProviderWithPatientDetailsdata shape-',Test_PatientDetail_labeldata.shape)
# print('Train_ProviderWithPatientDetailsdata shape-',Train_PatientDetail_labeldata.shape)

Train_PatientDetail_labeldata.rename({'PotentialFraud':'ClaimStatus'},inplace=True)
# print(Train_PatientDetail_labeldata.columns)

###Impute numeric columns with 0
cols1 = Train_PatientDetail_labeldata.select_dtypes([np.number]).columns
cols2 = Train_PatientDetail_labeldata.select_dtypes(exclude = [np.number]).columns

Train_PatientDetail_labeldata[cols1] = Train_PatientDetail_labeldata[cols1].fillna(value=0)
Test_PatientDetail_labeldata[cols1]=Test_PatientDetail_labeldata[cols1].fillna(value=0)

## FEATURE SELECTION
###Remove unnecessary columns
cols=Train_PatientDetail_labeldata.columns
cols[:58]
remove_these_columns=['BeneID', 'ClaimStartDt','ClaimEndDt','AttendingPhysician',
       'OperatingPhysician', 'OtherPhysician', 'ClmDiagnosisCode_1',
       'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4',
       'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7',
       'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10',
       'ClmProcedureCode_1', 'ClmProcedureCode_2', 'ClmProcedureCode_3',
       'ClmProcedureCode_4', 'ClmProcedureCode_5', 'ClmProcedureCode_6',
       'ClmAdmitDiagnosisCode',
        'DOB', 'DOD',
        'State', 'County']
Train_data=Train_PatientDetail_labeldata.drop(axis=1, columns=remove_these_columns)
Test_data=Test_PatientDetail_labeldata.drop(axis=1,columns=remove_these_columns)
# print('Train_data shape', Train_data.shape)
# print('Test_data shape', Test_data.shape)

##Type conversion
###Convert gender & race to categorical
Train_data.Gender=Train_data.Gender.astype('category')
Test_data.Gender=Test_data.Gender.astype('category')

Train_data.Race=Train_data.Race.astype('category')
Test_data.Race=Test_data.Race.astype('category')

###Dummification
Train_data=pd.get_dummies(Train_data,columns=['Gender','Race'],drop_first=True)
Test_data=pd.get_dummies(Test_data,columns=['Gender','Race'],drop_first=True)

###Convert Target 1(Yes) and 0(No)
Train_data.PotentialFraud.replace(['Yes','No'],['1','0'],inplace=True)
print(Train_data['PotentialFraud'].head())

###Data aggregration to providers level
Train_data_groupedbyProv_PF=Train_data.groupby(['ClaimID','PotentialFraud'],as_index=False).agg('sum')
Test_data_groupedbyProv_PF=Test_data.groupby(['ClaimID'],as_index=False).agg('sum')
# print(Train_data_groupedbyProv_PF.head())

##Train Validation Split
X=Train_data_groupedbyProv_PF.drop(axis=1,columns=['ClaimID','PotentialFraud'])
y=Train_data_groupedbyProv_PF['PotentialFraud']
# print(X.head())

##Standardization
###Apply StandardScaler() and transform values to its z form (-3 to 3)
sc=StandardScaler(copy=True, with_mean=False, with_std=False) #normalized (0,1)  # MinMaxScaler
sc.fit(X)
X_std=sc.transform(X)

X_teststd=sc.transform(Test_data_groupedbyProv_PF.iloc[:,1:])
# print(X_std)


# from sklearn.feature_selection import SelectKBest
# # from sklearn.feature_selection import f_classif
# from sklearn.feature_selection import chi2

# bestfeatures = SelectKBest(score_func=chi2, k=10)
# # bestfeatures = f_classif(X, y)
# fit = bestfeatures.fit(X_std, y)

# dfscores = pd.DataFrame(fit.scores_)
# dfcolumns = pd.DataFrame(X_std.columns)

# #concat two dataframes for better visualization 
# featureScores = pd.concat([dfcolumns,dfscores],axis=1)
# featureScores.columns = ['Specs','Score']  #naming the dataframe columns
# print(featureScores.nlargest(10,'Score'))  #print 10 best features

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
fvalue_selector = SelectKBest(f_classif, k=30)
X_kbest = fvalue_selector.fit_transform(X_std, y)


dfscores = pd.DataFrame(fvalue_selector.scores_)
dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
# print(featureScores.nlargest(30,'Score'))
##visualize
# dset = pd.DataFrame()
# dset['attr'] = X.columns
# dset['importance'] = fvalue_selector.estimator_.feature_importances_

# dset = dset.sort_values(by='importance', ascending=False)


# plt.figure(figsize=(16, 14))
# plt.barh(y=dset['attr'], width=dset['importance'], color='#1976D2')
# plt.title('RFECV - Feature Importances', fontsize=20, fontweight='bold', pad=20)
# plt.xlabel('Importance', fontsize=14, labelpad=20)
# plt.show()

##Split data in train and validation
### 'stratify=y' will make sure equal distribution of yes:no in both train and validation
X_train,X_val,y_train,y_val = train_test_split(X_kbest,y,test_size=0.3,random_state=101,stratify=y,shuffle=True)
# print(X_train)
# print(y_train)
# print(y_val)
from imblearn.combine import SMOTETomek
sm=SMOTETomek()
X_Train,y_Train=sm.fit_sample(X_train, y_train)
# print('X_train without SmoteTomek',X_train.shape)
# print('X_Train with SmoteTomek', X_Train.shape)
# print(y_Train)

##Model Building

###LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegressionCV
log=LogisticRegressionCV(Cs=10, fit_intercept=True, cv=10, dual=False, penalty='l2', scoring=None, solver='lbfgs', tol=1e-4, max_iter=10000, class_weight='balanced', n_jobs=None, verbose=0, refit=True, intercept_scaling=1.,random_state=123)
# log = LogisticRegressionCV(cv=10,class_weight='balanced',random_state=123)
log.fit(X_Train,y_Train)

###SVM
# from sklearn.svm import LinearSVC
# log=LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)
# log.fit(X_Train,y_Train)

# ### Lets predict probability of 0 and 1 for X_train and X_val
log_train_pred_probability=log._predict_proba_lr(X_Train)
log_val_pred_probability=log._predict_proba_lr(X_val)
# print(log_train_pred_probability[0:5])

# ## Lets Set probability Threshold to 0.50

log_train_pred_60=(log._predict_proba_lr(X_Train)[:,1]>0.50).astype(bool)
log_val_pred_60=(log._predict_proba_lr(X_val)[:,1]>0.50).astype(bool)   # set threshold as 0.50
# print(log_train_pred_60[0:5])

##coefficients of LR
pd.DataFrame(zip(X.columns, np.transpose(log.coef_)))[0:5]

# # ROC curve
from sklearn.metrics import roc_curve, auc,precision_recall_curve
fpr, tpr, thresholds = roc_curve(y_val,log._predict_proba_lr(X_val)[:,1])         #log_val_pred_probability[:,1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')

for label in range(1,10,1):
    plt.text((10-label)/10,(10-label)/10,thresholds[label*15],fontdict={'size': 14})

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

##Confusion matrix, Accuracy, sensitivity and specificity

from sklearn.metrics import confusion_matrix,accuracy_score,cohen_kappa_score,roc_auc_score,f1_score,auc

cm0 = confusion_matrix(y_Train, log_train_pred_60,labels=[1,0])         
print('Confusion Matrix Train : \n', cm0)

cm1 = confusion_matrix(y_val, log_val_pred_60,labels=[1,0])
print('Confusion Matrix Val: \n', cm1)

total0=sum(sum(cm0))
total1=sum(sum(cm1))
#####from confusion matrix calculate accuracy
accuracy0=(cm0[0,0]+cm0[1,1])/total0
print ('Accuracy Train: ', accuracy0)

accuracy1=(cm1[0,0]+cm1[1,1])/total1
print ('Accuracy Val: ', accuracy1)
sensitivity0 = cm0[0,0]/(cm0[0,0]+cm0[0,1])
print('Sensitivity Train : ', sensitivity0 )

sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
print('Sensitivity Val: ', sensitivity1 )


specificity0 = cm0[1,1]/(cm0[1,0]+cm0[1,1])
print('Specificity Train: ', specificity0)

specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
print('Specificity Val: ', specificity1)

KappaValue=cohen_kappa_score(y_val, log_val_pred_60)
print("Kappa Value :",KappaValue)
AUC=roc_auc_score(y_val, log_val_pred_60)

print("AUC:",AUC)

print("F1-Score Train: ",f1_score(y_Train, log_train_pred_60))

print("F1-Score Val: ",f1_score(y_val, log_val_pred_60))

