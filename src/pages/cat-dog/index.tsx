import * as tf from "@tensorflow/tfjs";
import { Button, Card, List, Select, Tag } from "antd";
import { useState } from "react";

// 类型定义
type ImageData = {
  path: string;
  element: HTMLImageElement;
  tensor: tf.Tensor3D;
  label: number; // 0: 猫，1: 狗
};

type TrainingLog = {
  epoch: number;
  loss: number;
  accuracy: number;
};

type PredictionResult = {
  label: string;
  confidence: number;
};

// 预处理图片函数（组件外部的工具函数）
const preprocessImage = (img: HTMLImageElement): HTMLCanvasElement => {
  const canvas = document.createElement("canvas");
  canvas.width = 128;
  canvas.height = 128;

  const ctx = canvas.getContext("2d");
  if (!ctx) return canvas;

  // 保持比例缩放并居中裁剪
  const ratio = Math.min(128 / img.width, 128 / img.height);
  const newWidth = img.width * ratio;
  const newHeight = img.height * ratio;

  ctx.drawImage(
    img,
    (128 - newWidth) / 2,
    (128 - newHeight) / 2,
    newWidth,
    newHeight
  );
  return canvas;
};

export const CatDogPage = () => {
  // 状态管理
  const [model, setModel] = useState<tf.Sequential | null>(null);
  const [trainingLogs, setTrainingLogs] = useState<TrainingLog[]>([]);
  const [testSamples, setTestSamples] = useState<ImageData[]>([]);
  const [selectedTestIndex, setSelectedTestIndex] = useState<number>(-1);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [isTraining, setIsTraining] = useState(false);

  // 加载图片函数
  const loadImage = async (
    category: "cat" | "dog",
    i: number
  ): Promise<ImageData> => {
    const imgPath = `src/pages/cat-dog/image/${category}/${category}.${i}.jpg`;
    const img = new Image();
    img.src = imgPath;

    await new Promise((resolve, reject) => {
      img.onload = () => resolve(img);
      img.onerror = reject;
    });

    return {
      path: imgPath,
      element: img,
      tensor: tf.browser.fromPixels(preprocessImage(img)).div(255), // 归一化
      label: category === "cat" ? 0 : 1,
    };
  };

  // 加载数据集
  const loadDataset = async () => {
    const images: ImageData[] = [];
    const categories = ["cat", "dog"] as const;

    for (const category of categories) {
      for (let i = 1000; i < 1500; i++) {
        try {
          const imgData = await loadImage(category, i);
          images.push(imgData);
        } catch (error) {
          console.error(
            `加载${
              category === "cat" ? "猫" : "狗"
            }图片失败: ${category}.${i}.jpg`,
            error
          );
        }
      }
    }

    return images;
  };

  // 创建卷积神经网络模型
  const createCNNModel = () => {
    const model = tf.sequential({
      layers: [
        // 最大池化层：降低特征图尺寸，增强特征鲁棒性
        tf.layers.maxPooling2d({
          inputShape: [128, 128, 3], // 输入形状 [高度, 宽度, 通道数]
          poolSize: 2, // 池化窗口尺寸 2x2
          strides: 2, // 滑动步长：每次移动 n 像素，使输出尺寸减小到原先的 1/n
        }),

        // 卷积层：用于提取图像局部特征
        tf.layers.conv2d({
          filters: 32, // 卷积核数量，决定输出特征图的深度
          kernelSize: 3, // 卷积核尺寸 3x3
          activation: "relu", // 激活函数：修正线性单元，解决梯度消失问题
          padding: "same", // 边缘填充方式：保持输出尺寸与输入相同
        }),
        // 展平层：将多维特征图转换为一维向量
        tf.layers.flatten(),

        // 全连接层（输出层）：进行最终分类
        tf.layers.dense({
          units: 2, // 输出单元数：对应猫/狗两个类别
          activation: "softmax", // 激活函数：将输出转换为概率分布
        }),
      ],
    });

    // 编译模型：配置训练参数
    model.compile({
      optimizer: "adam",
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"],
    });

    console.log("模型架构:");
    model.summary();

    return model;
  };

  // 准备训练数据
  const prepareTrainingData = (images: ImageData[]) => {
    const tensors = images.map((img) => img.tensor);
    return {
      xData: tf.stack(tensors) as tf.Tensor4D,
      yData: tf.oneHot(
        tf.tensor1d(
          images.map((img) => img.label),
          "int32"
        ),
        2
      ) as tf.Tensor2D,
    };
  };

  // 训练模型
  const trainModel = async (
    model: tf.Sequential,
    xData: tf.Tensor4D,
    yData: tf.Tensor2D
  ) => {
    setTrainingLogs([]); // 清空之前的训练日志

    await model.fit(xData, yData, {
      epochs: 10,
      batchSize: 4,
      validationSplit: 0.4,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          if (!logs) return;
          setTrainingLogs((prev) => [
            ...prev,
            {
              epoch: epoch + 1,
              loss: Number(logs.loss.toFixed(4)),
              accuracy: Number(logs.acc.toFixed(4)),
            },
          ]);
        },
      },
    });
  };

  // 预测函数
  const predict = async (imgData: ImageData) => {
    if (!model) return;

    const input = imgData.tensor.expandDims(0); // 添加批次维度
    const prediction = model.predict(input) as tf.Tensor;
    const probabilities = Array.from(prediction.dataSync());

    setPrediction({
      label: probabilities[0] > probabilities[1] ? "猫" : "狗",
      confidence: Math.max(...probabilities) * 100,
    });

    tf.dispose([input, prediction]);
  };

  // 初始化训练过程
  const initializeTraining = async () => {
    try {
      setIsTraining(true);
      setTrainingLogs([]);

      const allImages = await loadDataset();

      // 分割训练集和测试集
      const shuffledImages = [...allImages];
      tf.util.shuffle(shuffledImages);

      const testSet = shuffledImages.slice(-10); // 取最后4个作为测试集
      const trainSet = shuffledImages.slice(0, -10);

      setTestSamples(testSet);

      const { xData, yData } = prepareTrainingData(trainSet);
      const newModel = createCNNModel();
      await trainModel(newModel, xData, yData);
      setModel(newModel);

      // 训练完成后，默认选择第一个测试样本
      if (testSet.length > 0) {
        setSelectedTestIndex(0);
        predict(testSet[0]);
      }
    } catch (error) {
      console.error("训练初始化失败:", error);
    } finally {
      setIsTraining(false);
    }
  };

  // 选择测试样本
  const handleTestSampleChange = (index: number) => {
    setSelectedTestIndex(index);
    predict(testSamples[index]);
  };

  // 组件渲染
  return (
    <div className="max-w-[1200px] mx-auto p-6">
      <div className="mb-8">
        <Button
          type="primary"
          size="large"
          onClick={initializeTraining}
          loading={isTraining}
        >
          {model ? "重新训练模型" : "开始训练"}
        </Button>
        {model && (
          <Tag color="green" className="!ml-4">
            模型已就绪
          </Tag>
        )}
      </div>

      <Card title="训练进度">
        {trainingLogs.length === 0 ? (
          <div className="text-center text-gray-500">尚未开始训练</div>
        ) : (
          <List
            bordered
            dataSource={trainingLogs}
            renderItem={(log) => (
              <List.Item>
                <div className="flex justify-between w-full">
                  <span>Epoch {log.epoch}</span>
                  <span>Loss: {log.loss}</span>
                  <span>Accuracy: {(log.accuracy * 100).toFixed(1)}%</span>
                </div>
              </List.Item>
            )}
          />
        )}
      </Card>

      <Card title="模型测试" className="!mt-[20px]">
        <div className="mb-4">
          <span className="mr-4">测试样本:</span>
          <Select
            value={selectedTestIndex}
            onChange={handleTestSampleChange}
            style={{ width: 200 }}
            disabled={!testSamples.length}
          >
            {testSamples.map((_, index) => (
              <Select.Option key={index} value={index}>
                测试样本 {index + 1} ({_.label === 0 ? "猫" : "狗"})
              </Select.Option>
            ))}
          </Select>
        </div>

        {selectedTestIndex !== -1 && testSamples[selectedTestIndex] && (
          <div className="mt-4">
            <div className="flex gap-6">
              <div>
                <img
                  src={testSamples[selectedTestIndex].element.src}
                  alt="测试样本"
                  className="w-32 h-32 object-cover"
                />
              </div>
              <div>
                <div className="text-lg">
                  真实类别:{" "}
                  {testSamples[selectedTestIndex].label === 0 ? "猫" : "狗"}
                </div>
                {prediction && (
                  <div className="mt-2">
                    <Tag color={prediction.label === "猫" ? "blue" : "gold"}>
                      预测结果: {prediction.label}
                    </Tag>
                    <div className="mt-2">
                      置信度: {prediction.confidence.toFixed(1)}%
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
};
