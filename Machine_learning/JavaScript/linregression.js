import * as tf from '@tensorflow/tfjs';
import abstract_regression from "./abstract-regression";

export default class LinearRegressionModel extends abstract_regression {
    initModelVariables() {
        this.k = tf.scalar(Math.random()).variable();
        this.b = tf.scalar(Math.random()).variable();
    }

    f = x => x.mul(this.k).add(this.b);
}
