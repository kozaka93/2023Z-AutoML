[Link do skryptu w colabie](https://colab.research.google.com/drive/1po2zJg6xmElZBlku9sVN2zxlFNSG-GPz?usp=sharing)

https://github.com/automl/auto-sklearn/issues/1684

    # 1. uninstall all affected packages
    !pip uninstall -y Cython scipy pyparsing scikit_learn imbalanced-learn mlxtend yellowbrick
    # 2. install packages to be downgraded
    !pip install Cython==0.29.36 scipy==1.9 pyparsing==2.4
    # 3. install older scikit-learn disregarding its dependencies
    !pip install scikit-learn==0.24.2 --no-build-isolation
    # 4. finally install auto-sklearn
    !pip install auto-sklearn
    # 5. then, try load 
