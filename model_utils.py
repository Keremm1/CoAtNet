from keras_tuner import BayesianOptimization
from utils import plot_acc

from models import CoAtNetHyperModel
from customcallbacks import ValACCEarlyStopping


def return_best_model(input_shape, num_classes, train_set, val_set, search_epochs):
    tuner = BayesianOptimization(
        hypermodel=CoAtNetHyperModel(num_classes),
        objective="val_accuracy", #objective vall acc ve acc arasında en büyük değişimi de baz alabilir
        max_trials=50,
        directory="searching",
        project_name="coatnet_search",
        overwrite=True
    )

    tuner.search_space_summary()

    tuner.search(*train_set, epochs=search_epochs, validation_data=val_set,
                callbacks=[ValACCEarlyStopping(monitor='val_accuracy', baseline_metric='accuracy', patience=3)])
    models = tuner.get_best_models(num_models=2)
    
    best_model = models[0]

    best_model.build(input_shape=(None, *input_shape))
    best_model.summary()

    tuner.results_summary()

    return best_model


def build_model(model, input_shape, optimizer, loss, metrics, **kwargs):
    model =  model(**kwargs)
    model.build((None, *input_shape))
    model.summary()
    model.compile(
            optimizer=optimizer,
            loss=loss, #from_logits, label_smoothing
            metrics=metrics)
    return model


def evaluate_model(model, history, x_test, y_test):
    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    plot_acc(train_accuracy, val_accuracy, accuracy, top_5_accuracy, save_name="CoatNet")