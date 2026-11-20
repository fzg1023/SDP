"""
ACGAN-FG（Shao et al.）迁移到 NEU-Gear，按本文统一协议对比。

忠实复用其核心架构与训练损失：
  autoencoder(MLP) + generator + discriminator(WGAN-GP) + classifier(属性回归) + comparator(相似度)。
  输入样本 = 原始振动信号(2048)，语义属性 = 本文 21 维图像域物理属性（连续值，MSE 回归，
  原 TEP 为 20 维二值 BCE）。

协议与本文 SDP 完全一致（数据由 prep_acgan_fg_data.py 生成，npz 直接加载）：
  20% 标定集算类均值属性原型、80% ZSL、留二类(5 对)、seen 训练 / unseen 测试、3 种子(42/49/56)。
  测试：用 unseen 类原型属性驱动 generator 生成特征，训练 2 类下游分类器，在真实 unseen
  特征上测 2 类准确率（不早停：只用训练结束后的最终模型评估）。

用法：
  D:/Anaconda/envs/tf-env/python.exe -u acgan_fg_on_gear.py --epochs 100 --seeds 3
  D:/Anaconda/envs/tf-env/python.exe -u acgan_fg_on_gear.py --quick     # 冒烟测试(1对1种子2epoch)
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Input, Dense, LeakyReLU, ReLU, LayerNormalization,
                                     BatchNormalization, Conv1D, Reshape, Flatten, Concatenate,
                                     concatenate, multiply)
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mean_squared_error
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_NPZ = os.path.join(HERE, "acgan_fg_gear_data.npz")

CLASS_NAMES = ["健康", "剥落", "断齿", "磨损", "裂纹"]
ALL_PAIRS = [("健康", "裂纹"), ("剥落", "磨损"), ("断齿", "磨损"),
             ("剥落", "裂纹"), ("剥落", "断齿")]

DATA_LENGTH = 2048          # 原始信号长度（原 TEP=52）
FEATURE_DIM = 256           # autoencoder 潜在特征维
ATTR_DIM = 21               # 连续属性维（原 TEP=20 二值）
LATENT_DIM = 50
NUM_CLASSES = 5
LAMBDA_GP = 10
LAMBDA_CLA = 10
LAMBDA_CMS = 10
N_CRITIC = 1
ADD_QUANTITY = 2000         # 每 unseen 类生成特征数（原 2000）


class RandomWeightedAverage(Concatenate):
    """真实/生成特征之间的随机加权插值（WGAN-GP 梯度惩罚用）。"""
    def call(self, inputs):
        batch_size = tf.shape(inputs[0])[0]
        alpha = K.random_uniform((batch_size, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


class ZeroShot:
    def __init__(self):
        self.data_length = DATA_LENGTH
        self.sample_shape = (DATA_LENGTH,)
        self.feature_dim = FEATURE_DIM
        self.feature_shape = (FEATURE_DIM,)
        self.attr_dim = ATTR_DIM
        self.latent_dim = LATENT_DIM
        self.noise_shape = (LATENT_DIM, 1)
        self.n_critic = N_CRITIC
        self.LAMBDA_GP = LAMBDA_GP
        self.lambda_cla = LAMBDA_CLA
        self.lambda_cms = LAMBDA_CMS

        self.autoencoder_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.c_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.m_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        self.autoencoder = self.build_autoencoder()
        self.d = self.build_discriminator()
        self.g = self.build_generator()
        self.c = self.build_classifier()
        self.m = self.build_comparator()

        sample_1 = Input(shape=self.sample_shape)
        feature_1, _ = self.autoencoder(sample_1)
        sample_2 = Input(shape=self.sample_shape)
        feature_2, _ = self.autoencoder(sample_2)
        predicted_similarity = self.m([feature_1, feature_2])
        self.m_model = Model(inputs=[sample_1, sample_2], outputs=[predicted_similarity])
        self.m_model.compile(loss=['binary_crossentropy'], optimizer=self.m_optimizer)

    def build_autoencoder(self):
        sample = Input(shape=self.sample_shape)
        a1 = LeakyReLU(alpha=0.2)(LayerNormalization()(Dense(100)(sample)))
        a2 = LeakyReLU(alpha=0.2)(LayerNormalization()(Dense(200)(a1)))
        feature = LeakyReLU(alpha=0.2)(LayerNormalization()(Dense(256)(a2)))
        a4 = LeakyReLU(alpha=0.2)(LayerNormalization()(Dense(200)(feature)))
        a5 = LeakyReLU(alpha=0.2)(LayerNormalization()(Dense(100)(a4)))
        output_sample = LeakyReLU(alpha=0.2)(LayerNormalization()(Dense(self.data_length)(a5)))
        return Model(sample, [feature, output_sample])

    def build_discriminator(self):
        sample_input = Input(shape=self.feature_shape)
        reshaped_sample = Reshape((self.feature_dim, 1))(sample_input)
        attribute = Input(shape=(self.attr_dim,), dtype='float32')
        reshape_attribute = Reshape((self.attr_dim, 1))(attribute)
        concatenated = concatenate([reshaped_sample, reshape_attribute], axis=1)
        d1 = LeakyReLU(alpha=0.2)(LayerNormalization()(Dense(200)(Flatten()(concatenated))))
        d2 = LeakyReLU(alpha=0.2)(LayerNormalization()(Dense(100)(d1)))
        validity = Dense(1)(d2)
        return Model([sample_input, attribute], validity)

    def build_generator(self):
        noise = Input(shape=self.noise_shape)
        attribute = Input(shape=(self.attr_dim,), dtype='float32')
        reshape_attribute = Reshape((self.attr_dim, 1))(attribute)
        concatenated = concatenate([noise, reshape_attribute], axis=1)
        g1 = LeakyReLU(alpha=0.2)(LayerNormalization()(Dense(100)(Flatten()(concatenated))))
        g2 = LeakyReLU(alpha=0.2)(LayerNormalization()(Dense(200)(g1)))
        generated_feature = BatchNormalization()(LeakyReLU(alpha=0.2)(Dense(256)(g2)))
        return Model([noise, attribute], generated_feature)

    def build_classifier(self):
        sample = Input(shape=self.feature_shape)
        c1 = LeakyReLU(alpha=0.2)(Dense(100)(sample))
        c2 = LeakyReLU(alpha=0.2)(Dense(50)(c1))
        predict_attribute = Dense(self.attr_dim)(c2)  # 连续属性 → 线性 + MSE
        return Model(sample, [c2, predict_attribute])

    def build_comparator(self):
        def conv_block(x, filters, stride, kernel_size=3):
            x = Conv1D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            return x
        sample_1 = Input(shape=self.feature_shape)
        sample_2 = Input(shape=self.feature_shape)
        s1 = Reshape((self.feature_dim, 1))(sample_1)
        s2 = Reshape((self.feature_dim, 1))(sample_2)
        c0 = concatenate([s1, s2], axis=-1)
        c1 = conv_block(c0, filters=16, stride=2)
        c2 = conv_block(c1, filters=32, stride=2)
        c3 = conv_block(c2, filters=64, stride=1)
        c4 = Flatten()(c3)
        c5 = LeakyReLU(alpha=0.2)(LayerNormalization()(Dense(1500)(c4)))
        c6 = LeakyReLU(alpha=0.2)(LayerNormalization()(Dense(100)(c5)))
        similarity = Dense(1, activation="sigmoid")(c6)
        return Model([sample_1, sample_2], similarity)

    # ---- 损失（原样保留，仅分类损失 BCE→MSE）----
    def gradient_penalty_loss(self, gradients):
        gradients_sqr = tf.square(gradients)
        gradients_sqr_sum = tf.reduce_sum(gradients_sqr, axis=tf.range(1, len(gradients_sqr.shape)))
        gradient_l2_norm = tf.sqrt(gradients_sqr_sum)
        gradient_penalty = tf.square(1 - gradient_l2_norm)
        return tf.reduce_mean(gradient_penalty)

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def classification_loss(self, y_true, pred_attribute):
        # 连续属性 → MSE（原为二值 BCE）
        return mean_squared_error(y_true, pred_attribute)


def scalar_stand(train_x, test_x):
    scalar_train = preprocessing.StandardScaler().fit(train_x)
    return scalar_train.transform(train_x), scalar_train.transform(test_x)


def train_pair(gan, class_samples, class_attrs, epochs, batch_size):
    """在 3 个 seen 类上训练 ACGAN-FG（class_samples/class_attrs 各 3 项）。"""
    n_cls = len(class_samples)
    train_X = np.concatenate(class_samples, axis=0).astype(np.float32)
    train_Y = np.concatenate(class_attrs, axis=0).astype(np.float32)
    cls_id = np.concatenate([np.full(len(class_samples[k]), k) for k in range(n_cls)])
    boundaries = np.cumsum([0] + [len(c) for c in class_samples])

    num_batches = train_X.shape[0] // batch_size
    valid = -np.ones((batch_size, 1), dtype=np.float32)
    fake = np.ones((batch_size, 1), dtype=np.float32)
    similar_truth = np.ones((batch_size, 1), dtype=np.float32)
    unsimilar_truth = np.zeros((batch_size, 1), dtype=np.float32)

    for epoch in range(epochs):
        for batch_i in range(num_batches):
            start_i = batch_i * batch_size
            end_i = start_i + batch_size
            train_x = train_X[start_i:end_i]
            train_y = train_Y[start_i:end_i]
            hom_cls = int(np.searchsorted(boundaries, start_i, side='right') - 1)

            # 1) autoencoder（重建 MSE）
            with tf.GradientTape(persistent=True) as tape_auto:
                feature, output_sample = gan.autoencoder(train_x)
                autoencoder_loss = mean_squared_error(train_x, output_sample)
            grads_autoencoder = tape_auto.gradient(autoencoder_loss, gan.autoencoder.trainable_weights)
            gan.autoencoder_optimizer.apply_gradients(zip(grads_autoencoder, gan.autoencoder.trainable_weights))
            del tape_auto

            # 2) classifier（属性 MSE）
            with tf.GradientTape(persistent=True) as tape_c:
                feature_c, _ = gan.autoencoder(train_x)
                hidden_c, predict_c = gan.c(feature_c)
                c_loss = gan.classification_loss(train_y, predict_c)
            grads_c = tape_c.gradient(c_loss, gan.c.trainable_weights)
            gan.c_optimizer.apply_gradients(zip(grads_c, gan.c.trainable_weights))
            del tape_c

            # 3) comparator（相似/不相似 BCE）
            m_loss_1 = gan.m_model.train_on_batch([train_x, train_x], [similar_truth])
            other_mask = cls_id != hom_cls
            other_X = train_X[other_mask]
            idx = np.random.choice(len(other_X), size=batch_size, replace=False)
            m_loss_2 = gan.m_model.train_on_batch([train_x, other_X[idx]], [unsimilar_truth])

            # 4) discriminator（WGAN-GP）
            for _ in range(gan.n_critic):
                with tf.GradientTape(persistent=True) as tape_d:
                    noise = tf.random.normal(shape=(batch_size, LATENT_DIM, 1))
                    fake_feature = gan.g([noise, train_y])
                    real_feature, _ = gan.autoencoder(train_x)
                    interpolated_feature = RandomWeightedAverage()([real_feature, fake_feature])
                    real_validity = gan.d([real_feature, train_y])
                    fake_validity = gan.d([fake_feature, train_y])
                    interploted_validity = gan.d([interpolated_feature, train_y])
                    d_loss_real = gan.wasserstein_loss(valid, real_validity)
                    d_loss_fake = gan.wasserstein_loss(fake, fake_validity)
                    gradients = tape_d.gradient(interploted_validity, interpolated_feature)
                    gradient_penalty = gan.gradient_penalty_loss(gradients)
                    d_loss = d_loss_real + d_loss_fake + gan.LAMBDA_GP * gradient_penalty
                grads_d = tape_d.gradient(d_loss, gan.d.trainable_weights)
                gan.d_optimizer.apply_gradients(zip(grads_d, gan.d.trainable_weights))
                del tape_d

            # 5) generator（对抗 + 属性 + 相似度）
            hetero_classes = [k for k in range(n_cls) if k != hom_cls]
            homogeneous_feature, _ = gan.autoencoder(train_x)
            heterogeneous_features, heterogeneous_attributes = [], []
            for k in hetero_classes:
                idxk = np.random.choice(len(class_samples[k]), size=batch_size, replace=True)
                fk, _ = gan.autoencoder(class_samples[k][idxk].astype(np.float32))
                heterogeneous_features.append(fk)
                heterogeneous_attributes.append(class_attrs[k][idxk].astype(np.float32))

            with tf.GradientTape(persistent=True) as tape_g:
                noise_g = tf.random.normal(shape=(batch_size, LATENT_DIM, 1))
                fake_feature_g = gan.g([noise_g, train_y])
                fake_validity_g = gan.d([fake_feature_g, train_y])
                adversarial_loss = gan.wasserstein_loss(valid, fake_validity_g)

                _, fake_cla_g = gan.c(fake_feature_g)
                cla_loss = gan.classification_loss(train_y, fake_cla_g)

                comparison_loss_simi = tf.keras.losses.binary_crossentropy(
                    similar_truth, gan.m([fake_feature_g, homogeneous_feature]))
                comparison_loss_unsimi = tf.reduce_sum([
                    tf.keras.losses.binary_crossentropy(
                        unsimilar_truth, gan.m([fake_feature_g, heterogeneous_features[i]]))
                    for i in range(len(hetero_classes))], axis=0)

                total_loss = (adversarial_loss + gan.lambda_cla * cla_loss
                              + gan.lambda_cms * comparison_loss_simi
                              + gan.lambda_cms * comparison_loss_unsimi)
            grads_g = tape_g.gradient(total_loss, gan.g.trainable_weights)
            gan.g_optimizer.apply_gradients(zip(grads_g, gan.g.trainable_weights))
            del tape_g

        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            print(f"    [epoch {epoch + 1}/{epochs}] done", flush=True)


def diagnose(gan, test_x, test_y, unseen_proto, add_quantity):
    """用 unseen 类原型属性生成特征，训练 2 类下游分类器，测真实 unseen 特征。"""
    generated, labels_train = [], []
    for j in range(2):
        attr = np.tile(unseen_proto[j:j + 1], (add_quantity, 1)).astype(np.float32)
        noise = tf.random.normal(shape=(add_quantity, LATENT_DIM, 1))
        g = gan.g.predict([noise, attr], verbose=0)
        generated.append(g)
        labels_train.append(np.full((add_quantity, 1), j))
    generated = np.concatenate(generated, axis=0)
    labels_train = np.concatenate(labels_train, axis=0).ravel()

    test_feature = gan.autoencoder.predict(test_x.astype(np.float32), verbose=0)[0]
    hidden_train = gan.c.predict(generated, verbose=0)[0]
    hidden_test = gan.c.predict(test_feature, verbose=0)[0]

    new_train = np.concatenate([generated, hidden_train], axis=1)
    new_test = np.concatenate([test_feature, hidden_test], axis=1)
    new_train, new_test = scalar_stand(new_train, new_test)

    clfs = {
        "svm": LinearSVC(),
        "rf": RandomForestClassifier(n_estimators=100),
        "nb": GaussianNB(),
        "mlp": MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam',
                             alpha=0.0001, batch_size='auto', learning_rate='constant',
                             max_iter=200, tol=0.0001, random_state=42),
    }
    accs = {}
    for name, clf in clfs.items():
        clf.fit(new_train, labels_train)
        accs[name] = float(accuracy_score(test_y, clf.predict(new_test)))
    return accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=120)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="冒烟测试：1 对 1 种子 2 epoch")
    ap.add_argument("--pair-idx", type=int, default=None,
                    help="只跑第 N 对（0-4），None=全部；用于并行分片")
    ap.add_argument("--results-json", type=str,
                    default=os.path.join(HERE, "results_acgan_fg_gear.json"))
    args = ap.parse_args()

    d = np.load(DATA_NPZ)
    signals_z = d["signals_z"]          # (N, 2048) 标准化信号
    attrs_z = d["attrs_z"]              # (N, 21) 标准化属性
    labels = d["labels"]                # (N,) 0..4
    zsl_pos = d["zsl_pos"]              # 80% ZSL 索引
    proto = d["proto"]                  # (5, 21) 类均值原型
    n2i = {n: i for i, n in enumerate(CLASS_NAMES)}

    if args.quick:
        pairs = ALL_PAIRS[:1]
        seeds = [42]
        epochs = 2
    else:
        pairs = ALL_PAIRS if args.pair_idx is None else [ALL_PAIRS[args.pair_idx]]
        seeds = [42 + i * 7 for i in range(args.seeds)]
        epochs = args.epochs

    rows = []
    for a_name, b_name in pairs:
        a, b = n2i[a_name], n2i[b_name]
        seen_classes = [n2i[n] for n in CLASS_NAMES if n not in (a_name, b_name)]
        zsl_labels = labels[zsl_pos]
        unseen_mask = (zsl_labels == a) | (zsl_labels == b)
        test_idx = zsl_pos[unseen_mask]
        test_x = signals_z[test_idx]
        test_y = np.where(labels[test_idx] == a, 0, 1)

        for s in seeds:
            np.random.seed(s)
            tf.random.set_seed(s)
            gan = ZeroShot()
            class_samples = [signals_z[zsl_pos[zsl_labels == c]] for c in seen_classes]
            class_attrs = [attrs_z[zsl_pos[zsl_labels == c]] for c in seen_classes]
            unseen_proto = proto[[a, b]]

            print(f"\n=== {a_name}/{b_name} | seen={[CLASS_NAMES[c] for c in seen_classes]} "
                  f"| seed {s} (n_train={sum(len(x) for x in class_samples)}, n_test={len(test_x)}) ===",
                  flush=True)
            train_pair(gan, class_samples, class_attrs, epochs, args.batch_size)

            accs = diagnose(gan, test_x, test_y, unseen_proto, ADD_QUANTITY)
            best = max(accs.values())
            rows.append({"pair": f"{a_name}/{b_name}", "seed": s, **{k: round(v, 4) for k, v in accs.items()},
                         "best": round(best, 4)})
            print(f"  [seed {s}] { {k: round(v * 100, 2) for k, v in accs.items()} } "
                  f"best={best * 100:.2f}%", flush=True)

    out = {"note": "ACGAN-FG 迁移到 NEU-Gear（统一协议：20% 标定原型 + 留二类 + seen 训练/unseen 测试，"
                   "样本=原始信号2048，属性=21维连续MSE，不早停）",
           "epochs": epochs, "batch_size": args.batch_size, "seeds": seeds,
           "add_quantity": ADD_QUANTITY, "rows": rows}
    # 汇总
    if rows:
        cls_names = ["svm", "rf", "nb", "mlp"]
        summary = {}
        for c in cls_names:
            summary[c] = round(float(np.mean([r[c] for r in rows])), 4)
        summary["best_mean"] = round(float(np.mean([r["best"] for r in rows])), 4)
        out["summary"] = summary
        print("\n=== 各下游分类器 5对×种子 均值 ===", flush=True)
        for c in cls_names:
            print(f"  {c}: {summary[c] * 100:.2f}%", flush=True)
        print(f"  best(每对每种子取最大再平均): {summary['best_mean'] * 100:.2f}%", flush=True)

    json.dump(out, open(args.results_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[已另存] {args.results_json}")


if __name__ == "__main__":
    main()
