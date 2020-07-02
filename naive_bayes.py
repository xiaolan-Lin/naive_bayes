from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import cross_val_score


def create_dataset():
    """
    加载鸢尾花数据集
    """
    iris_data = load_iris()
    feature = iris_data.data
    labels = iris_data.target
    return feature, labels


def gnb_model(feature, labels):
    """
    朴素贝叶斯--高斯分布型
    """
    gnb_model = GaussianNB()  # 构建高斯模型

    gnb_model.fit(feature, labels)
    pred = gnb_model.predict(feature)
    print("==============================")
    print("高斯分布型模型预测准确有", sum(pred == labels), "组特征")
    return gnb_model


def mnb_model(feature, labels):
    """
    朴素贝叶斯--多项式型
    """
    mnb_model = MultinomialNB()  # 构建高斯模型

    mnb_model.fit(feature, labels)
    pred = mnb_model.predict(feature)
    print("==============================")
    print("多项式型模型预测准确有", sum(pred == labels), "组特征")
    return mnb_model


def bnb_model(feature, labels):
    """
    朴素贝叶斯--伯努利型
    """
    bnb_model = BernoulliNB()  # 构建高斯模型

    bnb_model.fit(feature, labels)
    pred = bnb_model.predict(feature)
    print("==============================")
    print("伯努利型模型预测准确有", sum(pred == labels), "组特征")
    return bnb_model


def cross_val(model, feature, labels):
    score = cross_val_score(model, feature, labels, cv=10)
    print("交叉验证模型准确率为：%.3f" % score.mean())


if __name__ == "__main__":
    feature, labels = create_dataset()
    print("========================朴素贝叶斯=========================")
    print("原始数据集中有", len(feature), "组特征")
    gnb = gnb_model(feature, labels)
    cross_val(gnb, feature, labels)
    mnb = mnb_model(feature, labels)
    cross_val(mnb, feature, labels)
    bnb = bnb_model(feature, labels)
    cross_val(bnb, feature, labels)
