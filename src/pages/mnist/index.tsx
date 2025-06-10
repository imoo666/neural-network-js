import * as tf from "@tensorflow/tfjs";
import { Button, Card, List, Tag } from "antd";
import { useState } from "react";
import { InfoCard } from "../../components/InfoCard";
import { ResultTitle } from "../../components/ResultTitle";

// 类型定义
interface DigitSample {
  pixels: tf.Tensor3D;
  label: number;
}

interface TrainingLog {
  epoch: number;
  loss: number;
  accuracy: number;
}

interface PredictionResult {
  imageTensor: tf.Tensor3D;
  actual: number;
  predicted: number;
  confidence: number;
  correct: boolean;
}

export const MnistPage = () => {
  const [modelState, setModelState] = useState<{
    model: tf.Sequential | null;
    isTraining: boolean;
    logs: TrainingLog[];
  }>({ model: null, isTraining: false, logs: [] });

  const [predictions, setPredictions] = useState<PredictionResult[]>([]);

  // 读取数据与预处理
  const loadCsvData = async () => {
    const response = await fetch("src/pages/mnist/assets/mnist.csv");
    const text = await response.text();
    const lines = text.trim().split("\n").slice(1);
    const samples: DigitSample[] = lines.map((line) => {
      const values = line.split(",").map(Number);
      const label = values[0];
      const pixels = tf
        .tensor3d(values.slice(1), [28, 28, 1])
        .div(255) as tf.Tensor3D;
      return { pixels, label };
    });
    tf.util.shuffle(samples);
    const train = samples.slice(0, samples.length - 50);
    const test = samples.slice(-50);

    const xTrain = tf.stack(train.map((s) => s.pixels)) as tf.Tensor4D;
    const yTrain = tf.oneHot(
      train.map((s) => s.label),
      10
    ) as tf.Tensor2D;

    return { xTrain, yTrain, test };
  };

  // 定义模型
  const defineModel = () => {
    const model = tf.sequential({
      layers: [
        tf.layers.maxPooling2d({
          poolSize: 2,
          strides: 2,
          inputShape: [28, 28, 1],
        }),
        tf.layers.conv2d({
          filters: 32,
          kernelSize: 3,
          activation: "relu",
          padding: "same",
        }),
        tf.layers.flatten(),
        tf.layers.dense({ units: 64, activation: "relu" }),
        tf.layers.dropout({ rate: 0.3 }),
        tf.layers.dense({
          units: 10,
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
    const { xTrain, yTrain, test } = await loadCsvData();

    await model.fit(xTrain, yTrain, {
      epochs: 20,
      batchSize: 8,
      validationSplit: 0.2,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          if (!logs) return;
          setModelState((prev) => ({
            ...prev,
            logs: [
              ...prev.logs,
              {
                epoch: epoch + 1,
                loss: Number(logs.loss?.toFixed(4)),
                accuracy: Number((logs.acc ?? logs.accuracy ?? 0).toFixed(4)),
              },
            ],
          }));
        },
      },
    });

    predict(model, test);
    setModelState((prev) => ({ ...prev, model, isTraining: false }));
    tf.dispose([xTrain, yTrain]);
  };

  // 预测结果
  const predict = (model: tf.Sequential, samples: DigitSample[]) => {
    const results: PredictionResult[] = samples.map((sample) => {
      const input = sample.pixels.expandDims(0);
      const output = model.predict(input) as tf.Tensor;
      const probs = output.dataSync();
      const predicted = output.argMax(1).dataSync()[0];
      const confidence = Number((probs[predicted] * 100).toFixed(1));
      tf.dispose([input, output]);
      return {
        imageTensor: sample.pixels,
        actual: sample.label,
        predicted,
        confidence,
        correct: predicted === sample.label,
      };
    });
    setPredictions(results);
  };

  // 渲染单张图片为 Base64
  const renderImage = (tensor: tf.Tensor3D): string => {
    const [w, h] = tensor.shape;
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    const imageData = ctx!.createImageData(w, h);
    const data = tensor.mul(255).dataSync();

    for (let i = 0; i < data.length; i++) {
      const value = data[i];
      imageData.data[i * 4 + 0] = value;
      imageData.data[i * 4 + 1] = value;
      imageData.data[i * 4 + 2] = value;
      imageData.data[i * 4 + 3] = 255;
    }
    ctx!.putImageData(imageData, 0, 0);
    return canvas.toDataURL();
  };

  return (
    <div className="max-w-[1200px] mx-auto p-6">
      <div className="flex gap-4 items-center mb-8">
        <Button
          type="primary"
          size="large"
          onClick={trainModel}
          disabled={modelState.isTraining}
        >
          {modelState.model ? "重新训练模型" : "开始训练"}
        </Button>
        {modelState.model && <Tag color="green">模型已就绪</Tag>}
      </div>

      <InfoCard
        className="!mb-6"
        title="手写数字识别"
        accuracy={90}
        rounds={20}
        totalTime={1}
      />
      <Card title="训练进度" className="!mb-6 max-h-[300px] overflow-y-auto">
        {modelState.logs.length > 0 ? (
          <List
            bordered
            dataSource={modelState.logs}
            renderItem={(log) => (
              <List.Item>
                <div className="flex justify-between w-full">
                  <div>轮数 {log.epoch}</div>
                  <div>损失: {log.loss}</div>
                  <div>准确率: {(log.accuracy * 100).toFixed(1)}%</div>
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

      <Card title={<ResultTitle predictions={predictions} />}>
        {modelState.model && predictions.length > 0 ? (
          <List
            bordered
            dataSource={predictions}
            renderItem={(item, idx) => (
              <List.Item style={{ color: item.correct ? "green" : "red" }}>
                <div className="flex items-center gap-4 w-full">
                  <img
                    src={renderImage(item.imageTensor)}
                    alt={`digit-${idx}`}
                    style={{ width: 56, height: 56 }}
                  />
                  <div>实际: {item.actual}</div>
                  <div>预测: {item.predicted}</div>
                  <div>置信度: {item.confidence}%</div>
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
