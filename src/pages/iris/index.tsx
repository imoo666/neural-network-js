import {
  CheckCircleOutlined,
  CloudDownloadOutlined,
  DownloadOutlined,
} from "@ant-design/icons";
import * as tf from "@tensorflow/tfjs";
import { Button, Card, List, Radio, Select, Tag } from "antd";
import { useState } from "react";

interface IrisData {
  features: number[];
  label: IrisSpecies;
}

type IrisSpecies = "setosa" | "versicolor" | "virginica";

// 在文件顶部添加类型定义和常量
const SPECIES: IrisSpecies[] = ["setosa", "versicolor", "virginica"];
const NUM_CLASSES = SPECIES.length;

// 提取模型配置常量
const MODEL_CONFIG = {
  hiddenUnits: 6,
  learningRate: 0.05,
  epochs: 100,
  batchSize: 32,
  validationSplit: 0.2,
};

// 提取数据解析函数
const parseIrisData = (data: string): IrisData[] => {
  return data
    .split(/\r?\n/)
    .filter((line) => line.trim())
    .map((line) => {
      const [sepalL, sepalW, petalL, petalW, species] = line.split(",");
      return {
        features: [sepalL, sepalW, petalL, petalW].map(Number),
        label: species.trim() as IrisSpecies,
      };
    });
};

// 提取Tensor转换函数
const convertToTensor = (data: IrisData[]) => ({
  xs: tf.tensor2d(data.map((d) => d.features)),
  ys: tf.oneHot(
    tf.tensor1d(
      data.map((d) => SPECIES.indexOf(d.label)),
      "int32"
    ),
    NUM_CLASSES
  ),
});

export const IrisPage = () => {
  const [model, setModel] = useState<tf.Sequential>();
  const [prediction, setPrediction] = useState<{
    label: string;
    confidence: number;
  }>();
  const [trainingLogs, setTrainingLogs] = useState<
    { epoch: number; loss: number; accuracy: number }[]
  >([]);
  const [testSamples, setTestSamples] = useState<IrisData[]>([]);
  const [selectedTestIndex, setSelectedTestIndex] = useState<number>(-1);
  const [loadedModel, setLoadedModel] = useState<tf.LayersModel>();
  const [isLoadingModel, setIsLoadingModel] = useState(false);
  const [selectedModelType, setSelectedModelType] = useState<
    "trained" | "loaded"
  >("trained");
  const [isTraining, setIsTraining] = useState(false);

  // 新增公共数据加载函数
  const loadAndPrepareData = async () => {
    const response = await fetch("src/pages/iris/assets/iris.txt");
    const data = await response.text();
    const parsedData = parseIrisData(data);

    tf.util.shuffle(parsedData);
    if (testSamples.length === 0) {
      setTestSamples(parsedData.slice(140));
      setSelectedTestIndex(0);
    }

    return parsedData;
  };

  // 修改后的loadData函数
  const loadData = async () => {
    const parsedData = await loadAndPrepareData();
    return convertToTensor(parsedData.slice(0, 140));
  };

  // 优化后的模型创建函数
  const createModel = () => {
    return tf.sequential({
      layers: [
        tf.layers.dense({
          units: MODEL_CONFIG.hiddenUnits,
          activation: "relu",
          inputShape: [4],
        }),
        tf.layers.dense({
          units: MODEL_CONFIG.hiddenUnits,
          activation: "relu",
        }),
        tf.layers.dense({
          units: MODEL_CONFIG.hiddenUnits,
          activation: "relu",
        }),
        tf.layers.dense({
          units: NUM_CLASSES,
          activation: "softmax",
        }),
      ],
    });
  };

  // 训练模型
  const trainModel = async () => {
    setIsTraining(true);
    setTrainingLogs([]);
    const model = createModel();

    model.compile({
      optimizer: tf.train.adam(MODEL_CONFIG.learningRate),
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"],
    });

    const train = await loadData();

    await model.fit(train.xs, train.ys, {
      epochs: MODEL_CONFIG.epochs,
      batchSize: MODEL_CONFIG.batchSize,
      validationSplit: MODEL_CONFIG.validationSplit,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          if (!logs) return;
          setTrainingLogs((prev) => [
            ...prev,
            {
              epoch: epoch + 1,
              loss: Number(logs.loss?.toFixed(4)) || 0,
              accuracy: Number(logs.acc?.toFixed(4)) || 0,
            },
          ]);
        },
      },
    });

    setModel(model);
    setIsTraining(false);
  };

  // 修改后的loadExternalModel函数
  const loadExternalModel = async () => {
    setIsLoadingModel(true);

    try {
      const model = await tf.loadLayersModel(
        "src/pages/iris/assets/iris-model.json"
      );
      const parsedData = await loadAndPrepareData();

      if (parsedData.length > 0) {
        predict(parsedData[140].features);
      }

      setLoadedModel(model);
      alert("模型加载成功！");
    } catch (error) {
      console.error("模型加载失败:", error);
    }
    setIsLoadingModel(false);
  };

  const predict = (features: number[], modelType = selectedModelType) => {
    const currentModel = modelType === "trained" ? model : loadedModel;
    if (!currentModel) return;

    const input = tf.tensor2d([features]);
    const prediction = currentModel.predict(input) as tf.Tensor; // 使用的是分类模型，所以得到了各个分类的概率
    const probabilities = Array.from(prediction.dataSync()); // 将概率转化为普通数组
    const predictedIndex = prediction.argMax(1).dataSync()[0]; // 获取概率最大的分类的索引

    setPrediction({
      label: SPECIES[predictedIndex],
      confidence: Math.round(probabilities[predictedIndex] * 100),
    });

    tf.dispose(input);
  };

  // 添加模型导出方法
  const exportModel = async () => {
    if (!model) return;

    // 保存模型为文件下载
    await model.save("downloads://iris-model");
  };

  return (
    <div className="max-w-[1200px] mx-auto p-6">
      <div className="flex flex-wrap gap-4 items-center mb-8">
        <Button
          type="primary"
          size="large"
          onClick={trainModel}
          icon={<CheckCircleOutlined />}
        >
          {model ? "重新训练模型" : "开始训练"}
        </Button>

        {model && (
          <>
            <Button
              type="default"
              size="large"
              onClick={exportModel}
              icon={<DownloadOutlined />}
            >
              导出模型
            </Button>
            <Tag color="green" icon={<CheckCircleOutlined />}>
              模型已就绪
            </Tag>
          </>
        )}

        <Button
          type="dashed"
          size="large"
          onClick={loadExternalModel}
          loading={isLoadingModel}
          icon={<CloudDownloadOutlined />}
        >
          加载外部模型
        </Button>
        {loadedModel && (
          <Tag color="purple" icon={<CheckCircleOutlined />} className="ml-4">
            外部模型已加载
          </Tag>
        )}
      </div>

      <Card title="训练进度">
        <div className="max-h-[300px] overflow-y-auto">
          {trainingLogs.length > 0 ? (
            <List
              bordered
              dataSource={trainingLogs}
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
              {isTraining ? "训练进行中..." : "训练尚未开始"}
            </div>
          )}
        </div>
      </Card>

      <Card title="模型测试" className="!mt-[20px]">
        <div className="flex flex-wrap gap-4 mb-6">
          <div className="flex-1 min-w-[300px]">
            <div className="mb-4">
              <span className="font-medium mr-2">选择测试模型:</span>
              <Radio.Group
                value={selectedModelType}
                onChange={(e) => {
                  const value = e.target.value;
                  setSelectedModelType(value);
                  if (testSamples.length > 0 && selectedTestIndex !== -1) {
                    predict(testSamples[selectedTestIndex].features, value);
                  }
                }}
              >
                <Radio
                  value="trained"
                  disabled={!model}
                  className="!h-10 !leading-10"
                >
                  自训练模型{model ? "" : "（未训练）"}
                </Radio>
                <Radio
                  value="loaded"
                  disabled={!loadedModel}
                  className="!h-10 !leading-10"
                >
                  外部模型{loadedModel ? "" : "（未加载）"}
                </Radio>
              </Radio.Group>
            </div>
          </div>

          <div className="flex-1 min-w-[300px]">
            <div className="mb-4">
              <span className="font-medium mr-2">选择测试样本:</span>
              <Select
                value={selectedTestIndex}
                onChange={(value) => {
                  setSelectedTestIndex(value);
                  if (testSamples.length > 0 && value !== -1) {
                    predict(testSamples[value].features, selectedModelType);
                  }
                }}
                style={{ width: "100%" }}
                disabled={!testSamples.length}
              >
                {testSamples.length > 0 ? (
                  testSamples.map((sample, index) => (
                    <Select.Option key={index} value={index}>
                      样本 #{index + 1} ({sample.label})
                    </Select.Option>
                  ))
                ) : (
                  <Select.Option value={-1}>请先开始训练</Select.Option>
                )}
              </Select>
            </div>
          </div>
        </div>

        {prediction && (
          <Card
            title={`预测结果（使用${
              selectedModelType === "trained" ? "自训练" : "外部"
            }模型）`}
            style={{
              borderLeft: `4px solid ${
                prediction.label === testSamples[selectedTestIndex].label
                  ? "#52c41a"
                  : "#ff4d4f"
              }`,
            }}
          >
            <div className="flex gap-4">
              <div className="flex-1">
                <div className="font-medium text-gray-600">模型预测</div>
                <div className="text-lg font-bold text-gray-800">
                  {prediction.label}
                </div>
              </div>
              <div className="flex-1">
                <div className="font-medium text-gray-600">实际品种</div>
                <div className="text-lg font-bold text-gray-800">
                  {testSamples[selectedTestIndex].label}
                </div>
              </div>
            </div>
            <div className="mt-4 text-right text-blue-500">
              置信度: {prediction.confidence}%
            </div>
          </Card>
        )}
      </Card>
    </div>
  );
};
