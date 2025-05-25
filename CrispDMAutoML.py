import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import ttest_rel
from tpot import TPOTClassifier, TPOTRegressor
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore")

class CrispDMAutoML:
    def __init__(self, dataset_path: str, class_target: str, reg_target: str):
        self.dataset_path = dataset_path
        self.class_target = class_target
        self.reg_target = reg_target
        self.df = None
        self.scaled_features = None
        self.class_models = {}
        self.reg_models = {}
        self.kmeans_model = None
    # Klasifikacija u kategorije potencijala (na temelju udjela OIE)
    @staticmethod
    def classify_potential(value):
        if value < 20:
            return "Nizak"
        elif value < 35:
            return "Srednji"
        else:
            return "Visok"
        
    # 1. Business Understanding
    def business_understanding(self):
        print("Cilj: Razviti CRISP-DM AutoML sustav koji klasificira, predviđa i segmentira regije po OIE potencijalu prema danim podacima sa Eurostata.")
        # Učitavanje izvornog Eurostat CSV-a
        df_raw = pd.read_csv("Plin2025Dataset/estat_nrg_ind_ren_filtered_en.csv")
        # Ekstrahiranje relevantnih stupaca
        df_clean = df_raw[["Geopolitical entity (reporting)", "TIME_PERIOD", "OBS_VALUE"]].copy()
        df_clean.columns = ["Country", "Year", "OIE_share"]
        # Uklanjanje redaka bez podataka
        df_clean.dropna(subset=["OIE_share"], inplace=True)
        
        df_clean["OIE_klasa"] = df_clean["OIE_share"].apply(CrispDMAutoML.classify_potential)
        # Spremanje pripremljene verzije CSV-a
        output_path = "Plin2025Dataset/eurostat_renewable_dataset.csv"
        df_clean.to_csv(output_path, index=False)
        print(f"Pripremljeni podaci spremljeni u {output_path}")




    # 2. Data Understanding
    def data_understanding(self):
        print("\nUčitavanje i razumijevanje podataka")
        self.df = pd.read_csv(self.dataset_path)
        print(self.df.info())
        print(self.df.describe())
        print("\nNedostajuće vrijednosti:")
        print(self.df.isnull().sum())

    # 3. Data Preparation
    def data_preparation(self):
        print("\nPriprema podataka: One-Hot Encoding + skaliranje značajki")

        # Učitavanje ako nije već učitano
        if self.df is None:
            print("Podaci nisu prethodno učitani. Učitavam iz:", self.dataset_path)
            self.df = pd.read_csv(self.dataset_path)

        # Brisanje redaka s nedostajućim vrijednostima
        self.df.dropna(inplace=True)

        # Odvajanje ulaznih značajki
        X = self.df.drop(columns=[self.class_target, self.reg_target])

        # One-Hot Encoding za nenumeričke značajke (npr. 'Country')
        X_encoded = pd.get_dummies(X, drop_first=True)

        # Log prikaz dummy varijabli
        encoded_columns = list(X_encoded.columns)
        print(f"One-Hot Encoding primijenjen. Ukupno {len(encoded_columns)} značajki:")
        print(encoded_columns)

        # Skaliranje svih značajki
        scaler = StandardScaler()
        self.scaled_features = scaler.fit_transform(X_encoded)

        print("Skaliranje i kodiranje uspješno završeno.")

    # 4. Modeling - Classification
    def modeling_classification(self):
        print("\nModeliranje: Klasifikacija s TPOT i RandomForest")
        X = self.scaled_features
        y = self.df[self.class_target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Random Forest
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        print("RandomForest Klasifikacija:\n", classification_report(y_test, y_pred_rf))

        # TPOT AutoML
        tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, scoring='f1_macro', random_state=42)
        tpot.fit(X_train, y_train)
        y_pred_tpot = tpot.predict(X_test)
        print("AutoML TPOT Klasifikacija:\n", classification_report(y_test, y_pred_tpot))

        # Matrica zabune
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', ax=axes[0], cmap='Blues')
        axes[0].set_title("Random Forest – Confusion Matrix")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("True")

        sns.heatmap(confusion_matrix(y_test, y_pred_tpot), annot=True, fmt='d', ax=axes[1], cmap='Greens')
        axes[1].set_title("TPOT AutoML – Confusion Matrix")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("True")
        plt.tight_layout()
        plt.show()

        # Statistička validacija
        f1_rf = cross_val_score(rf, X, y, cv=5, scoring='f1_macro')
        f1_tpot = cross_val_score(tpot.fitted_pipeline_, X, y, cv=5, scoring='f1_macro')

        # Vizualizacija F1 rezultata
        plt.figure(figsize=(8, 4))
        sns.barplot(data=pd.DataFrame({
            "Model": ["Random Forest"] * len(f1_rf) + ["TPOT AutoML"] * len(f1_tpot),
            "F1 Score": list(f1_rf) + list(f1_tpot)
        }), x="Model", y="F1 Score")
        plt.title("Usporedba F1 rezultata (CV)")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()

        t_stat, p_val = ttest_rel(f1_tpot, f1_rf)
        print(f"\nPaired t-test (TPOT vs RF): t={t_stat:.4f}, p={p_val:.4f}")
        if p_val < 0.05:
            print("Statistički značajna prednost AutoML modela.")
        else:
            print("Nema značajne razlike između modela.")
        
        self.class_models = {"rf": rf, "tpot": tpot}
        # Feature importance za Random Forest
        if hasattr(rf, "feature_importances_"):
            importances = rf.feature_importances_
            feature_names = list(self.df.drop(columns=[self.class_target, self.reg_target]).select_dtypes(include=['number']).columns)
            if "Country" in self.df.columns:
                # dodaj dummy varijable ako je korišten One-Hot Encoding
                encoded = pd.get_dummies(self.df.drop(columns=[self.class_target, self.reg_target]), drop_first=True)
                feature_names = list(encoded.columns)

            feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
            feat_df.sort_values(by="Importance", ascending=False, inplace=True)

            plt.figure(figsize=(10, 6))
            sns.barplot(data=feat_df.head(20), x="Importance", y="Feature")
            plt.title("Najvažnije značajke (Random Forest – Klasifikacija)")
            plt.tight_layout()
            plt.show()



    # 5. Modeling - Regression
    def modeling_regression(self):
        print("\nModeliranje: Regresija s TPOT i RandomForest")
        X = self.scaled_features
        y = self.df[self.reg_target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf = RandomForestRegressor(random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)

        tpot = TPOTRegressor(generations=5, population_size=20, verbosity=2, scoring='neg_root_mean_squared_error', random_state=42)
        tpot.fit(X_train, y_train)
        y_pred_tpot = tpot.predict(X_test)

        print(f"RandomForest Regresija:\nRMSE: {mean_squared_error(y_test, y_pred_rf, squared=False):.4f} | R2: {r2_score(y_test, y_pred_rf):.4f}")
        print(f"AutoML TPOT Regresija:\nRMSE: {mean_squared_error(y_test, y_pred_tpot, squared=False):.4f} | R2: {r2_score(y_test, y_pred_tpot):.4f}")

        # Scatter plot predikcija
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(y_test, y_pred_rf, alpha=0.7, label='RF')
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
        plt.title("RF Predikcija vs Stvarne vrijednosti")
        plt.xlabel("Stvarne vrijednosti")
        plt.ylabel("Predviđene")

        plt.subplot(1, 2, 2)
        plt.scatter(y_test, y_pred_tpot, alpha=0.7, label='TPOT', color='green')
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
        plt.title("TPOT Predikcija vs Stvarne vrijednosti")
        plt.xlabel("Stvarne vrijednosti")
        plt.tight_layout()
        plt.show()

        # Statistička validacija
        scores_rf = cross_val_score(rf, X, y, cv=5, scoring='neg_root_mean_squared_error')
        scores_tpot = cross_val_score(tpot.fitted_pipeline_, X, y, cv=5, scoring='neg_root_mean_squared_error')

        plt.figure(figsize=(8, 4))
        sns.boxplot(data=pd.DataFrame({
            "Model": ["RF"] * len(scores_rf) + ["TPOT"] * len(scores_tpot),
            "RMSE": list(-scores_rf) + list(-scores_tpot)
        }), x="Model", y="RMSE")
        plt.title("Usporedba RMSE rezultata (CV)")
        plt.tight_layout()
        plt.show()

        t_stat, p_val = ttest_rel(-scores_tpot, -scores_rf)
        print(f"\nPaired t-test (TPOT vs RF): t={t_stat:.4f}, p={p_val:.4f}")
        if p_val < 0.05:
            print("AutoML regresijski model značajno bolji.")
        else:
            print("Nema značajne razlike u regresijskim performansama.")

        self.reg_models = {"rf": rf, "tpot": tpot}
        # Feature importance za Random Forest Regressor
        if hasattr(rf, "feature_importances_"):
            importances = rf.feature_importances_
            feature_names = list(self.df.drop(columns=[self.class_target, self.reg_target]).select_dtypes(include=['number']).columns)
            if "Country" in self.df.columns:
                encoded = pd.get_dummies(self.df.drop(columns=[self.class_target, self.reg_target]), drop_first=True)
                feature_names = list(encoded.columns)

            feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
            feat_df.sort_values(by="Importance", ascending=False, inplace=True)

            plt.figure(figsize=(10, 6))
            sns.barplot(data=feat_df.head(20), x="Importance", y="Feature")
            plt.title("Najvažnije značajke (Random Forest – Regresija)")
            plt.tight_layout()
            plt.show()

    # 6. Modeling - Clustering
    def modeling_clustering(self, n_clusters=3):
        print(f"\nKlasteriranje (KMeans) u {n_clusters} klastera...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(self.scaled_features)
        self.df['Cluster'] = cluster_labels
        print(self.df[['Country', 'Year', 'Cluster']].groupby(['Cluster']).count())
        self.kmeans_model = kmeans

        # Broj članova po klasterima
        plt.figure(figsize=(8, 5))
        sns.countplot(data=self.df, x='Cluster', palette='Set2')
        plt.title("Broj uzoraka po klasteru")
        plt.xlabel("Klaster")
        plt.ylabel("Broj regija")
        plt.tight_layout()
        plt.show()

        # Prikaz PCA reduciranog prostora
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(self.scaled_features)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=cluster_labels, palette='Set1', s=60)
        plt.title("PCA projekcija klastera")
        plt.xlabel("PCA komponenta 1")
        plt.ylabel("PCA komponenta 2")
        plt.legend(title="Klaster")
        plt.tight_layout()
        plt.show()

    # 7. Deployment
    def deployment(self):
        print("\nIzvoz modela: TPOT klasifikacija i regresija")
        self.class_models["tpot"].export("tpot_pipeline_classification.py")
        self.reg_models["tpot"].export("tpot_pipeline_regression.py")
        print("Modeli spremni za integraciju u produkcijske sustave.")

# === POKRETANJE ===
if __name__ == "__main__":
    crisp = CrispDMAutoML(
        dataset_path="Plin2025Dataset/eurostat_renewable_dataset.csv",
        class_target="OIE_klasa",
        reg_target="OIE_share"
    )
    crisp.business_understanding()
    crisp.data_understanding()
    crisp.data_preparation()
    crisp.modeling_classification()
    crisp.modeling_regression()
    crisp.modeling_clustering(n_clusters=3)
    crisp.deployment()

