可以在cmd输入下面这一句话。可以得到决策树的pdf可视化
dot -Tpdf F:\AllWorkSpace\PycharmWorkspace\DecisionTree\allElectronicInformationGainOri.dot -o F:\AllWorkSpace\PycharmWorkspace\DecisionTree\allElec
tronicInformationGainOri.pdf

前提是安装了 Graphviz： http://www.graphviz.org/

      配置环境变量 配置到bin文件夹
 转化dot文件至pdf可视化决策树：dot -Tpdf iris.dot -o outpu.pdf