import * as tf from "@tensorflow/tfjs";
import { Button, Card, List, Tag } from "antd";
import { useState } from "react";

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

type Prediction = {
  path: string;
  label: string;
  confidence: number;
  correct: boolean;
  actual: string;
};

const preprocessImage = (img: HTMLImageElement): HTMLCanvasElement => {
  const canvas = document.createElement("canvas");
  canvas.width = 128;
  canvas.height = 128;

  const ctx = canvas.getContext("2d");
  if (!ctx) return canvas;

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
  // 集中管理模型状态、训练状态、训练日志
  const [modelState, setModelState] = useState<{
    model: tf.Sequential | null;
    isTraining: boolean;
    logs: TrainingLog[];
  }>({
    model: null,
    isTraining: false,
    logs: [],
  });

  // 测试样本和预测结果
  const [testState, setTestState] = useState<{
    samples: ImageData[];
    predictions: Prediction[];
  }>({
    samples: [],
    predictions: [],
  });

  // 加载单张图片
  const loadImage = async (
    category: "cat" | "dog",
    i: number
  ): Promise<ImageData> => {
    const path = `src/pages/cat-dog/image/${category}/${category}.${i}.jpg`;
    const element = new Image();
    element.src = path;
    await new Promise<void>((resolve, reject) => {
      element.onload = () => resolve();
      element.onerror = () => reject(new Error(`加载图片失败：${path}`));
    });
    return {
      path,
      element,
      tensor: tf.browser.fromPixels(preprocessImage(element)).div(255),
      label: category === "cat" ? 0 : 1,
    };
  };

  // 加载全部数据集
  const loadAndPrepareData = async () => {
    const images: ImageData[] = [];
    const categories = ["cat", "dog"] as const;

    for (const category of categories) {
      for (let i = 1000; i < 1500; i++) {
        try {
          const imgData = await loadImage(category, i);
          images.push(imgData);
        } catch (e) {
          // 图片加载失败可忽略或打印日志
          console.warn(e);
        }
      }
    }

    tf.util.shuffle(images);
    const trainSamples = images.slice(0, images.length - 50);
    const testSamples = images.slice(-50);

    setTestState((prev) => ({ ...prev, samples: testSamples }));

    const xData = tf.stack(trainSamples.map((d) => d.tensor)) as tf.Tensor4D;
    const yData = tf.oneHot(
      tf.tensor1d(
        trainSamples.map((d) => d.label),
        "int32"
      ),
      2
    ) as tf.Tensor2D;

    return { xData, yData, testSamples };
  };

  // 定义模型
  const defineModel = () => {
    const model = tf.sequential({
      layers: [
        tf.layers.maxPooling2d({
          inputShape: [128, 128, 3],
          poolSize: 2,
          strides: 2,
        }),
        tf.layers.conv2d({
          filters: 32,
          kernelSize: 3,
          activation: "relu",
          padding: "same",
        }),
        tf.layers.flatten(),
        tf.layers.dense({
          units: 2,
          activation: "softmax",
        }),
      ],
    });

    model.compile({
      optimizer: "adam",
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"],
    });

    return model;
  };

  // 训练模型
  const trainModel = async () => {
    setModelState({ model: null, isTraining: true, logs: [] });

    const model = defineModel();
    const { xData, yData, testSamples } = await loadAndPrepareData();

    await model.fit(xData, yData, {
      epochs: 10,
      batchSize: 4,
      validationSplit: 0.4,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          if (!logs) return;
          setModelState((prev) => ({
            ...prev,
            logs: [
              ...prev.logs,
              {
                epoch: epoch + 1,
                loss: Number(logs.loss.toFixed(4)),
                accuracy: Number((logs.acc ?? logs.accuracy ?? 0).toFixed(4)),
              },
            ],
          }));
        },
      },
    });

    // 训练完成后预测测试集
    predictSamples(model, testSamples);

    setModelState((prev) => ({ ...prev, model, isTraining: false }));

    // 释放训练数据内存
    tf.dispose([xData, yData]);
  };

  // 预测多个样本
  const predictSamples = (model: tf.Sequential, samples: ImageData[]) => {
    const labels = ["猫", "狗"];

    const predictions = samples.map((sample) => {
      const input = sample.tensor.expandDims(0);
      const output = model.predict(input) as tf.Tensor;
      const probs = Array.from(output.dataSync());
      const index = output.argMax(1).dataSync()[0];
      tf.dispose([input, output]);

      return {
        path: sample.path,
        label: labels[index],
        confidence: Number((probs[index] * 100).toFixed(1)),
        correct: labels[index] === labels[sample.label],
        actual: labels[sample.label],
      };
    });

    setTestState((prev) => ({ ...prev, predictions }));
  };

  return (
    <div className="max-w-[1200px] mx-auto p-6">
      <div className="flex flex-wrap gap-4 items-center mb-8">
        <Button
          type="primary"
          size="large"
          onClick={trainModel}
          disabled={modelState.isTraining}
        >
          {modelState.model ? "重新训练模型" : "开始训练"}
        </Button>
        {modelState.model && (
          <Tag color="green" className="font-semibold">
            模型已就绪
          </Tag>
        )}
      </div>

      <Card title="训练进度" className="!mb-6 max-h-[300px] overflow-y-auto">
        {modelState.logs.length > 0 ? (
          <List
            bordered
            dataSource={modelState.logs}
            renderItem={(log) => (
              <List.Item>
                <div className="flex justify-between w-full gap-4">
                  <div className="flex-1">轮数 {log.epoch}</div>
                  <div className="flex-1">损失: {log.loss}</div>
                  <div className="flex-1">
                    准确率: {(log.accuracy * 100).toFixed(1)}%
                  </div>
                </div>
              </List.Item>
            )}
          />
        ) : (
          <div className="p-4 text-gray-500">
            {modelState.isTraining ? "训练进行中..." : "训练尚未开始"}
          </div>
        )}
      </Card>

      <Card title="预测结果">
        {modelState.model && testState.predictions.length > 0 ? (
          <List
            bordered
            dataSource={testState.predictions}
            renderItem={(item) => (
              <List.Item style={{ color: item.correct ? "green" : "red" }}>
                <div className="flex justify-between w-full gap-4 items-center">
                  <div className="flex-1">
                    <img
                      src={item.path}
                      alt="测试样本"
                      style={{ width: 64, height: 64, objectFit: "cover" }}
                    />
                  </div>
                  <div className="flex-1">实际: {item.actual}</div>
                  <div className="flex-1">预测: {item.label}</div>
                  <div className="flex-1">置信度: {item.confidence}%</div>
                </div>
              </List.Item>
            )}
          />
        ) : (
          <div className="p-4 text-gray-500">请先训练模型</div>
        )}
      </Card>
    </div>
  );
};
