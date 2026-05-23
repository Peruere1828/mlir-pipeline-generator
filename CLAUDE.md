这是一个mlir pass pipeline的自动搜索器，原理是把mlir代码视作operation和type的集合，然后通过启发式搜索来搜索可能的从源代码到目标dialect（这里是llvm方言）的lowering pipeline。

当下存在如下问题：
1. 对全局pass的支持比较差
2. pass仍然不能自动导入。可以考虑利用ai，阅读tablegen或者cpp源代码来导入
3. 不能复刻mlir的带有region等的精确的operation的语义
4. 目前理论上可以搜索出一些简单的pass，但是尚且没有在真实的测试集上做测试

可能的解决方案：
1. 改进对mlir代码的建模，甚至可以考虑不使用python，改用cpp来利用原生的API
2. 利用AI工具，比如可以利用http://127.0.0.1:8000的AI端口，使用anthropic协议向他请求，使用的模型名称为`deepseek-v4-flash`或`deepseek-v4-pro`。详细内容在~/projects/combine_docs/proxy.py展示。
3. 在mlir官方的测试集上进行测试。

你的目标：
1. 尽可能多地通过mlir官方测试集中对linalg及更底层方言如scf、cf、arith降级的相关测试
2. 尽可能地引入高效的pass刻画方案、搜索方案、pass导入方案

你可以使用的工具：
conda activate testenv
cmake *
make *
bash *
python3 *
git 
等等。

llvm-project的根目录在/home/ubuntuaaa/projects/mlir/llvm-project，版本22.1.x，已经build过了
环境变量里虽然也有mlir-opt、llvm-lit等，但是它是20.1.8版本的。注意与上述区分。

每当完成一个阶段的开发，使用 `git add *` 和 `git commit -m *`命令提交结果。这里*是通配符。