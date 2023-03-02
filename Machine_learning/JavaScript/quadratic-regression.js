import * as tf from '@tensorflow/tfjs';
import abstract_regression from "./abstract-regression";

export default class QuadraticRegressionModel extends AbstractRegressionModel {
    initModelVariables() {
        this.a = tf.scalar(Math.random()).variable();
        this.b = tf.scalar(Math.random()).variable();
        this.c = tf.scalar(Math.random()).variable();
    }

    f = x => this.a.mul(x.square()).add(this.b.mul(x)).add(this.c);
}
