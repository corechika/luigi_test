import pandas as pd
import numpy as np
import luigi
from sklearn import cluster

class MakeTestData(luigi.Task):
    n = luigi.IntParameter()

    def requires(self):
        pass

    def output(self):
        return luigi.LocalTarget('./luigi_test_df.txt')

    def run(self):
        df = pd.DataFrame()
        for i in range(10):
            df['col_'+str(i)] = np.random.randn(self.n)
        with self.output().open('w') as f:
            df.to_csv(f, sep='\t', index=False)

class DataPrePro(luigi.Task):
    n = luigi.IntParameter()

    def requires(self):
        return MakeTestData(n=self.n)

    def output(self):
        return luigi.LocalTarget('./luigit_test_after_prepro.txt')
    
    def sum_columns(self, row):
        return sum(np.sqrt(row.values**2))
    
    def run(self):
        with self.input().open('r') as fin, self.output().open('w') as fout:
            df = pd.read_csv(fin.name, sep='\t')
            df['sum'] = df.apply(lambda row: self.sum_columns(row), axis=1)
            df.to_csv(fout, sep='\t', index=False)
    

class TrainData(luigi.Task):
    k = luigi.IntParameter()
    n = luigi.IntParameter()

    def requires(self):
        return DataPrePro(n=self.n)
    
    def output(self):
        return luigi.LocalTarget('./kmeans_result.txt')
    
    def run(self):
        with self.input().open('r') as fin, self.output().open('w') as fout:
            df = pd.read_csv(fin, sep='\t')
            kmeans = cluster.KMeans(n_clusters=self.k)
            kmeans.fit(df.drop(['sum'], axis=1).values)
            pre = kmeans.predict(df.drop(['sum'], axis=1).values)
            df['cluster'] = pre

            df.to_csv(fout, sep='\t', index=False)
    
if __name__ == '__main__':
    luigi.run()