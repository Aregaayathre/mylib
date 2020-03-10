

import pandas as pd
import numpy as np


pd.Series()

ser = pd.Series([1,2,3,4])
ser
type(ser)


ser.name = 'Age'
type(ser.values)
ser.values

ser.index
type(ser.index)

ser1=pd.Series([1,2,3,4],index=["a","b","c","d"])

ser1.index

ser2=pd.Series([1,2,3,4],index=range(0,8,2))

ser2.index

ser1=pd.Series([1,2,3,4],index=[1,3,5,7])

ser1.index

ser1=pd.Series([1,2,3,4],index=[0,2,4,7])

ser1.index


ll=ser1.reindex(['a'])
ser1.reindex([1])


ser1
ser1.values
ser1.index

' The data can also come from dict , array or scalar . '

# dict

sample_dict = {'a':1,'b':2,'c':3,'d':4}

ps=pd.Series(sample_dict)

ps=pd.Series(sample_dict,index=['a','b','c'])
type(ps)

ps=pd.Series(sample_dict,index=['a','b','c','e'])



# array
tt=pd.Series(np.array([1,2,4]),index=np.array(['a','a','c']))
tt.reindex(['a'])

'# also to note index can also be non-unique but certain 
'# operations may not allow where an exception would be raised.

#
kk=pd.Series(5)

#

### reindex ---> Subsetting(Selection)

sample_dict

pd.Series(sample_dict,index=['a','b','e','b'])  # reindex

a = pd.Series(sample_dict,index=['a','b','e'])
a
type(a['e'])

'# the NaN is used for missing values - as a standard.

################################################################
' Series are similar to Numpy Arrays ' 

hh=np.exp(ser1)

type(ser1)

type(hh)

ser1

ser1[:3]

ser1>2

ser1[ser1>2]

ser1*2

np.exp(ser1)

' Series are similar to DICT ' 

ser1["a"]

ser1[["a","b"]]

'a' in sample_dict  # key- check
3 in sample_dict  # value-check

sample_dict['a']

'2' in ser1
2 in ser1
'a' in ser1

ser1.get('a')

gg=ser1.get('a')
hh=gg+2
print(hh)

gg=ser1.get('e')
print(gg)
hh=gg+2
print(hh)

print(gg)

ser1.get('e',"nothing returned")

ser1[1:]+ser1[:-1]   # mask  #intersection # cancellation

set1={'b','c','d'}  
set2={'a','b','c'}

set1.union(set2) - set1 

set1.union(set2) - set2  
##############################################################

dict1={"Name":"jill","Age":23,"City":"Blore"}

serr= pd.Series(dict1)

serr

serr1= pd.Series(dict1,index=["Name","Age","City","Hello"])

serr1

type(serr1['Hello'])

type(serr1['Name'])

type(serr1['Age'])

'Thus a dict of Series plus a specific index will discard all data not matching up to the passed index.'

sum(serr1.isnull())  # only for Series

pd.isnull(serr1)

sum(pd.notnull(serr1))

serr2=serr1.fillna(21)

serr1.fillna(21, inplace =True)

serr1.dropna()

################################################################

'DataFrame is a 2-dimensional labeled data structure with columns of different types.'

'''
Dict of 1D ndarrays, lists, dicts, or Series
2-D numpy.ndarray
Structured or record ndarray
A Series
Another DataFrame
'''

import pandas as pd
import numpy as np

data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'], 
        'year': [2000, 2001, 2002, 2001, 2002], 
       'pop': [1.5, 1.7, 3.6, 2.4, 2.9]} 

data

frame = pd.DataFrame(data) 

frame

type(frame)

frame.name = "Home"

frame

frame.name

frame.year

frame.state

type(frame.year)

frame1 = pd.DataFrame(data,columns=['year','pop','state','ee']) 

frame1

frame2 = pd.DataFrame(data,columns=['year','pop','state'],index=['one','two','three','four','five']) 

frame2

###################################################################

data = {'state':
    pd.Series(['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],index=['a','b','c','d','e']), 
    'year': 
        pd.Series([2000, 2001, 2002, 2001, 2002],index=['a','b','c','d','e']), 
    'pop':
        pd.Series([1.5, 1.7, 3.6, 2.4, 2.9],index=['a','b','c','d','e'])} 

data

frame2 = pd.DataFrame(data,columns=['year','pop','state'],index=['a','b','c','d']) 

frame2

frame2.index

frame2.columns

###################################################################

' # Column Selection , Addition and Deletion '

frame
type(frame)

frame.year
frame.state

frame[  'year'   ]
type(frame['year'])

frame[['year','pop']]
frame['year','pop']  
type(frame[  ['year','pop']  ])

frame.ndim
frame.shape

frame.year
type(frame.year)

frame.head()  # n = 5

frame['new_col'] = frame.year * 2 
frame['new_col_bool'] = frame.year > 2001 

frame.head()


del frame['new_col_bool']

del frame[['new_col','new_col_bool']] #invalid

frame.head()

frame['new_col'] = frame.year ** 2 
frame['new_col_bool'] = frame.year > 2001 

series_op = frame.pop('new_col')

series_op = frame.pop()

series_op

series_op = frame.pop()  # needs one posit.argument



data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'], 
        'year': [2000, 2001, 2002, 2001, 2002], 
       'pop': [1.5, 1.7, 3.6, 2.4, 2.9]} 

data

frame = pd.DataFrame(data) 

frame['new_cal']=frame.year*2
series_op=frame.pop('new_cal')


series_op1 = pd.concat([series_op,frame.pop('year')],axis=1)

series_op2 = pd.concat([series_op,frame.pop('year')],axis=0)

series_op


series_op = pd.concat([series_op,frame.pop('state')],axis=0)

type(series_op)

' # Function application and mapping '

import numpy as np
import pandas as pd

df = pd.DataFrame(np.random.randn(8, 4))
df

type(df)

np.exp(df)  # numpy functions retain the df-structure.

type(np.exp(df))

type(np.exp(df[0]))

jj=np.exp(df[0])


### UDF 

# iterate over any part.dimension

df.apply(np.exp)  #axis = 0 ---  columns

df.apply(np.exp,axis=0)

df.apply(np.exp,axis=1)

df.apply(np.exp)  #axis = 0 ---  columns

df.apply(np.exp,axis=0)

df.apply(np.exp,axis=1)


df.apply(some_func,axis=1) # rows

df.applymap(np.exp)  # apply + map

df.map(np.exp)

df[df.columns[1]].map(np.exp)

df[df.columns[1],2]

df[df.columns[1]].map(np.exp)



#

# Write a UDF to find diff between max-min for each columns
df.apply(max)

df.apply(max,axis=1)

df.apply(max,axis=0)


df.apply(max)-df.apply(min)

 df.apply(lambda x:x.max()-x.min(),axis=1)
 
 df.apply(lambda x:x.max()-x.min(),axis=0)




'#############################  Indexing #################################'
           
'''
Operation	                        Syntax	               Result

Select column	                    df[col]	               Series
Slice rows	                        df[5:10]	           DataFrame
Select rows by boolean vector	    df[bool_vec]           DataFrame


Select row by label	                df.loc[label]      	   Series
Select row by integer location      df.iloc[loc]	       Series

.loc .iloc .ix

'''
'##########################################################################'

                           ' Slicing and Dicing ' 
df.columns= range(1,5)
df.columns = ['one','two','three','four']

df[3]      #select
df['three']
df.three
df.'3'  # select
df[0:2] # slice
df[0:2]# slice

df.columns =['A','B','C','D']

df['A']
df[0:2]
df[0:2]


df[['A','B']]

df[['A','E']] # Exception raised.

df[[True,False]*4]

df[:2]

df[0::2]

df[   :2    ,    'A'   ]  # unhashable.


df[   :2]    ['A']

df[   :2][['A','B']]

df[   :2]['A','B']


df[   :2    ,    'A'   ]

#df.index = ['A','B','C','D','e','f','g','h']
#df.index= range(0,8)

df[[1:2,'A']]

df[3,'A']

df[1:5]

df[[True,False]*4,'A']

df[[True,False]*4]['B']

df[[1:2]['B']]

df.columns



df['A']

type(df)

import seaborn.apionly as sns

iris = sns.load_dataset('iris')

iris.head(10)

iris.tail(15)

iris['sepal_length']

iris[['sepal_length','petal_length']]

iris[['sepal_length','blabla']].head() 

iris[:3]      # interger [] slices on rows.

iris[:3,:2]   # unhashable slice

type(iris[['sepal_length','petal_length']])

type(iris['sepal_length'])

iris[[True,False]*75]  # rows

iris[[True,False]*75,['sepal_length','petal_length']] # will not give result

iris[[True,False]*75][['sepal_length','petal_length']]

iris[[True,False]*75]['sepal_length']


type(iris)

iris.columns


'''
# .ix supports mixed integer and label based access. 
# It is primarily label based, but will fall back to 
# integer positional access unless the corresponding axis is of 
# integer type.
'''

Label Indexing - Character based indexing.
Positional Based  - Number based indexing. # positional based. n:m-1

.ix   -  Label andd Postional based # deprecated. 

.loc  -  integers , char , boolean (Label Based Indexing)

.iloc -  integers . boolean  (Positional Based Indexing always)

array[np.ix_(      )   ] # numbers.

'####################################################################'

from pandas import date_range
import numpy as np
pd.__version__
import pandas as pd

pd.__version__

dates = date_range('1/1/2000', periods=8)

df = pd.DataFrame(np.random.randn(8, 4), index=[40,41,43,44,3,4,5,6], columns=['A', 'B', 'C', 'D'])
df

df.index

df.loc[:3]  # labels

df.loc[:43]  # labels


df.iloc[:3] # postions  # EXCLUSIVE :END:   n:m   m -exclusive

df.iloc[:1] # postions  # EXCLUSIVE :END:   n:m   m -exclusive


'##  .ix  .loc  .iloc  -------- MIXED datatypes ####'

df = pd.DataFrame(np.random.randn(8, 4), index=['a','b','c',44,3,4,5,6], columns=['A', 'B', 'C', 'D'])
df

df.index

 

df.loc[:'c']

df.iloc[:3]


'####################################################################'
import seaborn as sns

iris = sns.load_dataset('iris')

iris.head(10)

iris.ix[    : 4   ,    :  2  ]

iris.ix[  1  : 4   ,   1 :  2  ]

iris.iloc[  1  : 4   ,   1 :  2  ]

iris.iloc[  1  : 4   ,   'sepal_length'  ]

iris.iloc[ 1 : 4 ]

iris.loc[  1  : 4   ,   'sepal_length'  ]

iris.loc[  1  : 4   ,  1 : 2  ]


iris.ix[ 1:3 , 'sepal_length']

type(iris.ix[ 1:3 , 'sepal_length'])

iris.columns = [0,1,2,3,4]

iris.head(10)

iris.ix[    : 5   ,    :  3  ]

######################################################################

'''
# .iloc -.iloc is primarily integer position based 
#  (from 0 to length-1 of the axis), but may also be used with a 
# boolean array.

'''
iris = sns.load_dataset('iris')

iris.head(10)

iris.iloc[     :5 ,         :3]                                    #EXCLUSIVE OF THE :(END)  END.

iris.iloc[2,'petal_length']

iris.iloc[1:5,1:3]

iris.tail(10)

iris.iloc[-1,:]

iris.shape


abc  = list([1,2,3])
abc[1:1000000]

iris.tail(10)

iris.iloc[ -5:149 , : ]
iris.iloc[ -5:-1 , : ]

iris.iloc[ -5:149:-1 , : ]

iris.iloc[ -5:149:1 , : ]

iris.iloc[ -5: , : ]

'''
## .loc .loc is primarily label based, but may also be used with a 
boolean array
'''

iris.head()

iris.loc[:2,'petal_length']

iris.loc[ :5 , :2 ]

iris.iloc[ :5 , :2 ]


iris.ix[:2,'petal_length']

##########################################################################

iris.shape

iris.columns

iris.tail(20)

iris.index

# https://github.com/pandas-dev/pandas/issues/2600 -  The following are bugs.
iris.ix[  -10:   ,    :-4  ]
iris.ix[  :-10   ,    :-4  ]

iris.head()

iris.iloc[130:-5,4]

iris.iloc[130:-5,-4]

###########################################################################################
iris = sns.load_dataset('iris')

iris.head()

iris[   iris['sepal_length']>6   ]

iris[iris.sepal_length>3]


######################################################################

d={'1st col name':[1,2,3,4],'2nd col name':[5,6,7,8]}

newdf=pd.DataFrame(data=d,index=['a','b','c','d'])

newdf

newdf['1st col name']>2

# Masking

newdf.loc[   ['c','d']    ,      ]

newdf.loc[   newdf['1st col name']>2    , '2nd col name'     ]

newdf.loc[newdf['1st col name']>2,]=2000

type(newdf['1st col name']>2)


np.array(newdf['1st col name']>2).tolist()

newdf.iloc[[True,False]*2,:]



newdf.iloc[newdf['1st col name']>2, : ]
newdf.loc[newdf['1st col name']>2,  ]
newdf.loc[newdf['1st col name']>2, '2nd col name'  ]



newdf

# condtional subsetting. - sepal-length > 5 and petal length >1.5

iris.head(10)

iris['petal_length']>1.5
iris['sepal_length']>5
 
np.all([iris['sepal_length']>5,iris['petal_length']>1.5],axis=0)



iris[np.all([iris['sepal_length']>5,iris['petal_length']>1.5],axis=0)  ]

iris.loc[np.all([iris['sepal_length']>5,iris['petal_length']>1.5],axis=0), ]


#iloc wont work with boolean - see https://github.com/pandas-dev/pandas/issues/3631
# see http://stackoverflow.com/questions/16603765/what-is-the-most-idiomatic-way-
#     to-index-an-object-with-a-boolean-array-in-pandas

s = pd.Series(list('abcde'), index=[0,3,2,5,4])



iris.loc[[True,False]*75,[True,False,False,False,False]]

s

s.loc[3:5]
s.loc[1:6]
s.sort_index()
s.sort_index().loc[1:5]

############################   Reindexing ##############################

obj = pd.Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
obj

obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])
obj2

obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'],fill_value=0)
obj2

obj3 = pd.Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
obj3


obj3.reindex(range(6))

obj3.reindex(range(6), method='ffill') 

obj3.reindex(range(6), method='bfill') 




############################################################################

s = pd.Series(np.arange(5), index=np.arange(5)[::-1], dtype='int64')
s

s.isin(['2','4','6'])
s.isin([2,4,6]) # works for both

#how would you use to subset for values ?



#########################################################################
df<0 -  True

import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(8, 4), index=[40,41,43,44,3,4,5,6], columns=['A', 'B', 'C', 'D'])
df

df[df<0] 

df.where(df<0)

df1 = df.where(df<0,-999)

df.where(df<0,-999,inplace=True)

df



# inverse logic
df.mask(df<0)

df.mask(df<0,-999)

df.mask(df<0,-999,inplace=True)

#########################################################################

df = pd.DataFrame(np.random.rand(10, 3), columns=list('abc'))

df.query('a < b and b < c')

df.query('a<b<c')

df.query('a>b<c')
#########################################################################
import pandas as pd
import numpy as np

' ## Understanding Missing Values - NA '

df  = pd.DataFrame(np.random.rand(5, 3), columns=list('abc'))
df  = df.where(df>0.5)
df


df1 = pd.DataFrame(np.random.rand(5, 3), columns=list('abc'))
df1 = df1.where(df1>0.5)
df1



type(df.a[0])

df1.a[0]

df + df1


df['a'].sum()
df['a'].sum(skipna=False)  # when agg-func applied the NA's are equivalent to 0's
df['a'].mean()

df1.sum()

df1.sum(skipna=True)

df1.sum(skipna=False)


df.cumsum() # they are still preserved but ignored in calculations.

# np.nan '0'



'########## Handling NA s - Cleaning or Filling them ###############'

df.isnull().sum()

df.fillna('abc')

df.fillna(999)

df.fillna(-999)

df.fillna(method="ffill")

df.fillna(method="bfill")

some_dict = {'a':123,'b':345,'c':789}

df.fillna(some_dict)


######################################################################################


df = pd.DataFrame([[np.nan, 2, np.nan, 0], [3, 4, np.nan, 1],
                    [np.nan, np.nan, np.nan, 5]],
                   columns=list('ABCD'))
df

df.sum()

df.dropna(axis=0)  # column

df.dropna(axis=1)  # row

df.dropna(axis=0,how="any")

df.dropna(axis=1,how="any")

df.dropna(axis=0,how="all")

df.dropna(axis=1,how="all")



'#########################  Merging ############################'
import pandas as pd
import numpy as np

df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3'],
                     'C': ['C0', 'C1', 'C2', 'C3'],
                     'D': ['D0', 'D1', 'D2', 'D3']},
                     index=[0, 1, 2, 3])
 

df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                     'B': ['B4', 'B5', 'B6', 'B7'],
                    'C': ['C4', 'C5', 'C6', 'C7'],
                    'D': ['D4', 'D5', 'D6', 'D7']},)


df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
                   'B': ['B8', 'B9', 'B10', 'B11'],
                    'C': ['C8', 'C9', 'C10', 'C11'],
                  'E': ['D8', 'D9', 'D10', 'D11']},
                    index=[0, 1, 10, 11])

frames = [df1, df2, df3]


result = pd.concat(frames,axis=0,sort=False) # outer
result.head()
result.tail()

result = pd.concat(frames,axis=1,sort=False)
result.head()

result = pd.concat(frames,join="inner",axis=0)
result.head()

result = pd.concat(frames,join="inner",axis=1)
result.head()

result = pd.concat(frames,ignore_index=True,join="inner",axis=0)  # range()
result.head()
result.index

result.reset_index()

df1
df3

df1.append(df3) # columns not present are added as new columns
# pd.concat(axis=0)
' ############################## Single Series ############################ '

s1 = pd.Series(['X0', 'X1', 'X2', 'X3'], name='X')

s1

pd.concat([df1,s1],axis=0)

pd.concat([df1,s1],axis=1)

' ######################################################################### '

####################################################################################

#Creating df using import.
import os
import pandas as pd

os.chdir("D:\\Data Science\\")

os.getcwd()

# pwd

# read_kind
# to_kind

df = pd.read_csv("insurance.csv")

df.info()

df.describe()

df.columns

df.shape

df.head()

df.tail()

################################################################

df.corr()

df.cov()

################################################################

val=df['region']

pd.value_counts(val)


###

df=pd.read_csv("insurance.csv",header=0)

df.head()

df=pd.read_csv("insurance.csv",header=5)

df.head()

df=pd.read_csv("insurance.csv",header = None)

df.head()

df=pd.read_csv("insurance.csv",header=0,usecols=['age','sex','bmi'])

df.head()

df=pd.read_csv("insurance.csv",header=None,
               names=['a','b','c','d','e','f','g'])

df.head()

df=pd.read_csv("insurance.csv",header=None,
               names=['a','b','c','d','e','f','g'],na_values=["19","yes"])

df.head()

############################################################################

dict1={'a':[19,18],'f':'northwest','c':[27.9,33]}

df=pd.read_csv("insurance.csv",header=None,
               names=['a','b','c','d','e','f','g'],na_values=dict1)

df.head()

data = pd.read_table("insurance.csv")   #"tab sep"
data

data = pd.read_table("insurance.csv",sep=",")   #"tab sep"
data

data = pd.read_excel("Sales.xlsx")
data


'#### Exporting ####'


output="op.csv"

df.to_csv(output)

output="op.tsv"

df.to_csv(output,sep='\t')

output="op.xlsx"

df.to_excel(output)

##################################
# importing pandas module 
import pandas as pd 

# making data frame from csv file 
nba = pd.read_csv("nba.csv") 

nba 

#####################################
# importing pandas module 
import pandas as pd 

# making data frame from csv file 
nba = pd.read_csv("nba.csv") 

# replacing na values in college with No college 
nba["College"].fillna("No College", inplace = True) 

nba 
#####################################

# importing pandas module 
import pandas as pd 

# making data frame from csv file 
nba = pd.read_csv("nba.csv") 

# replacing na values in college with No college 
nba["College"].fillna( method ='ffill', inplace = True) 

nba 


# importing pandas module 
import pandas as pd 

# making data frame from csv file 
nba = pd.read_csv("nba.csv") 

# replacing na values in college with No college 
nba["College"].fillna( method ='ffill', limit = 1, inplace = True) 

nba 


# importing pandas module 
import pandas as pd 

# making data frame from csv file 
data = pd.read_csv("nba.csv") 

# making new data frame with dropped NA values 
new_data = data.dropna(axis = 0, how ='any') 

# comparing sizes of data frames 
print("Old data frame length:", len(data), "\nNew data frame length:", 
	len(new_data), "\nNumber of rows with at least 1 NA value: ", 
	(len(data)-len(new_data))) 
#############################

# importing pandas module 
import pandas as pd 

# making data frame from csv file 
data = pd.read_csv("nba.csv") 

# making a copy of old data frame 
new = pd.read_csv("nba.csv") 

# creating a value with all null values in new data frame 
new["Null Column"]= None

# checking if column is inserted properly 
print(data.columns.values, "\n", new.columns.values) 

# comparing values before dropping null column 
print("\nColumn number before dropping Null column\n", 
	len(data.dtypes), len(new.dtypes)) 

# dropping column with all null values 
new.dropna(axis = 1, how ='all', inplace = True) 

# comparing values after dropping null column 
print("\nColumn number after dropping Null column\n", 
	len(data.dtypes), len(new.dtypes)) 


