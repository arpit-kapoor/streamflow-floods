from __future__ import absolute_import

import numpy as np
import tensorflow as tf
import keras
import keras.backend as K

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
from keras.layers import Reshape
from keras.layers import Input
from keras.layers import Dropout
from keras.layers import Concatenate
from keras.layers import RepeatVector
from keras.layers import Flatten

from tensorflow.keras.optimizers import SGD

from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

from .window import WindowGenerator


class QuantileLoss():
    @staticmethod
    def quantile_loss(q, y_true, y_pred):
        e = (y_true - y_pred)
        return tf.reduce_mean(tf.maximum(q * e, (q - 1) * e))
    
    @staticmethod
    def qloss_05(y_true, y_pred):
        return QuantileLoss.quantile_loss(0.05, y_true, y_pred)
    
    @staticmethod
    def qloss_50(y_true, y_pred):
        return QuantileLoss.quantile_loss(0.5, y_true, y_pred)
    
    @staticmethod
    def qloss_70(y_true, y_pred):
        return QuantileLoss.quantile_loss(0.7, y_true, y_pred)
    
    @staticmethod
    def qloss_90(y_true, y_pred):
        return QuantileLoss.quantile_loss(0.9, y_true, y_pred)
    
    @staticmethod
    def qloss_95(y_true, y_pred):
        return QuantileLoss.quantile_loss(0.95, y_true, y_pred)

    
class Model():
    MAX_EPOCHS = 150
    
    def __init__(self, window):
        # Store the raw data.
        self.window = window
        
        self.train_df = self.window.train_df
        self.test_df = self.window.test_df
             
    def compile_and_fit(self, model, window, loss_func, patience=10):

        model.compile(loss=loss_func,
                    optimizer=tf.optimizers.Adam(),
                    metrics=[tf.metrics.MeanAbsoluteError()])

        print(f"  Training model for {self.MAX_EPOCHS} epochs...")
        history = model.fit(window.train, epochs=self.MAX_EPOCHS,
                            verbose=0)
        print(f"  Model training complete.")

        return history

    def num_flood_events(self, cut=1):
        actuals = np.squeeze(self.window.test_windows, axis=2) 
        cut_percentile = np.percentile(actuals.flatten(), cut)
        locs = np.unique(np.where(actuals<cut_percentile)[0])

        events = np.split(locs, np.cumsum( np.where(locs[1:] - locs[:-1] > 1) )+1)

        return len(events)
    
    def summary(self, station=None, **kwargs):
        summary_dict = {}
        
        summary_dict['model_name'] = self.model_name
        summary_dict['input_width'] = self.window.input_width
        summary_dict['label_width'] = self.window.label_width
             
        if station is None:
            summary_dict['station'] = self.window.station
            summary_dict['inputs'] = str(list(self.train_df.columns))
            summary_dict['NSE'] = self.get_NSE(**kwargs)
        else:
            summary_dict['station'] = station
            example_station = self.train_df.columns.get_level_values(0)[0]
            summary_dict['inputs'] = str(list(self.train_df[example_station].columns))
            summary_dict['NSE'] = self.get_NSE(station, **kwargs)  
                             
        summary_dict['SER_1%'] = self.average_model_error(station, cut=1, **kwargs)
        summary_dict['SER_2%'] = self.average_model_error(station, cut=2, **kwargs)    
        summary_dict['SER_5%'] = self.average_model_error(station, cut=5, **kwargs)        
        summary_dict['SER_10%'] = self.average_model_error(station, cut=10, **kwargs)  
        summary_dict['SER_25%'] = self.average_model_error(station, cut=25, **kwargs)  
        summary_dict['SER_50%'] = self.average_model_error(station, cut=50, **kwargs)  
        summary_dict['SER_75%'] = self.average_model_error(station, cut=75, **kwargs)  
        summary_dict['MSE'] = self.average_model_error(station, cut=100, **kwargs)
        

        # summary_dict['f1_score_individual_1%'] = self.binary_metrics(station=station, cut=1, metric='f1_score', evaluation='individual')        
        # summary_dict['f1_score_individual_2%'] = self.binary_metrics(station=station, cut=2, metric='f1_score', evaluation='individual')  
        # summary_dict['f1_score_individual_5%'] = self.binary_metrics(station=station, cut=5, metric='f1_score', evaluation='individual') 
        # summary_dict['f1_score_individual_10%'] = self.binary_metrics(station=station, cut=10, metric='f1_score', evaluation='individual') 
        # summary_dict['f1_score_individual_25%'] = self.binary_metrics(station=station, cut=25, metric='f1_score', evaluation='individual') 
        # summary_dict['f1_score_individual_50%'] = self.binary_metrics(station=station, cut=50, metric='f1_score', evaluation='individual')  
        # summary_dict['f1_score_individual_75%'] = self.binary_metrics(station=station, cut=75, metric='f1_score', evaluation='individual') 
        # summary_dict['f1_score_individual_all'] = self.binary_metrics(station=station, cut=100, metric='f1_score', evaluation='individual') 
          
        return summary_dict
            
    def print_model_error(self, station=None, cut=0):
        if station is not None:
            preds = self.predictions(station)
            actuals = self.window.test_windows(station)
            test_array = self.window.test_array(station)
        else:
            preds = self.predictions()
            actuals = self.window.test_windows()  
            test_array = self.window.test_array()
            
        cut_percentile = np.percentile(actuals.flatten(), cut)

        locs = np.unique(np.where(actuals>cut_percentile)[0])
        preds = preds[locs]
        actuals = actuals[locs]

        for window_pred, window_actual, loc in zip(preds, actuals, locs):
            print("time: {}".format(loc))
            print("Input: {}".format(test_array[loc:loc+self.window.input_width].flatten()))
            print("Predicted: {}".format(window_pred))
            print("Actual: {}".format(window_actual))
            print("-------------------------")
            
    def model_predictions_less_than_cut(self, cut=100):
        
        preds = self.predictions()
        actuals = self.window.test_windows()

        cut_percentile = np.percentile(actuals.flatten(), cut)

        num_predicted = (preds.flatten() < cut_percentile).sum()
        num_actual = (actuals.flatten() < cut_percentile).sum()

        return num_predicted, num_actual
        
    def average_model_error(self, station=None, cut=100, **kwargs):
        # if self.window.label_columns[0] == 'streamflow_MLd_inclInfilled':
        cut = 100 - cut

        if station is not None:
            preds = self.predictions(station)
            actuals = self.window.test_windows(station)
        else:
            preds = self.predictions()
            actuals = self.window.test_windows()

        # min and scale values for the station
        _min = kwargs.get('_min', None)
        _scale = kwargs.get('_scale', None)

        if _min is not None and _scale is not None:
            # print(f"station: {station}, cut: {cut}, _min: {_min}, _scale: {_scale}")
            preds = (preds - _min)/_scale
            actuals = (actuals - _min)/_scale

        cut_percentile = np.percentile(actuals.flatten(), cut)

        locs = np.where(actuals>cut_percentile)[0]
        preds = preds[locs]
        actuals = actuals[locs]

        avg_error = 0

        # print(f"preds shape: {preds.shape}, actuals shape: {actuals.shape}")

        for window_pred, window_actual in zip(preds, actuals):
            avg_error += np.sum((window_pred - window_actual)**2)
        


        if len(actuals.flatten()) == 0:
            print(f'station: {station}, cut: {cut}, len: {actuals.shape}')
            return np.nan
        else:
            avg_error = avg_error/(actuals.shape[0]*actuals.shape[1])
            return avg_error
    
    def get_NSE(self, station=None, return_type='cast', **kwargs):
        if station is not None:
            preds = self.predictions(station)
            actuals = self.window.test_windows(station)
        else:
            preds = self.predictions()
            actuals = self.window.test_windows()
        
        # min and scale values for the station
        _min = kwargs.get('_min', None)
        _scale = kwargs.get('_scale', None)

        if _min is not None and _scale is not None:
            preds = (preds - _min)/_scale
            actuals = (actuals - _min)/_scale
        
        NSE = []

        for i in range(self.window.label_width):
            numer = np.sum(np.square(preds[:, i] - actuals[:, i]))
            denom = np.sum(np.square(actuals[:, i] - np.mean(actuals[:, i])))

            print(f"station: {station} label: {i+1}, denom: {denom}, numer: {numer}")
        
            NSE.append(1-(numer/denom))
        
        if return_type == 'cast':
            return np.mean(NSE)
        else:
            return NSE

    def binary_metrics(self, cut, metric, evaluation='whole', station=None):
        percentile_cut = self.window.station_percentile(station=station, cut=cut)
        
        if station is None:
            preds_pre = self.predictions()
            actuals_pre = self.window.test_windows()
        else:        
            preds_pre = self.predictions(station)
            actuals_pre = self.window.test_windows(station)
            
        if evaluation == 'whole':
            preds = np.array([int(any(x > percentile_cut)) for x in preds_pre])
            actuals = np.array([int(any(x > percentile_cut)) for x in actuals_pre])
        else:
            preds = np.array([int(x > percentile_cut) for x in preds_pre.flatten()])           
            actuals = np.array([int(x > percentile_cut) for x in actuals_pre.flatten()])

        if metric == 'accuracy':
            return accuracy_score(actuals, preds)
        elif metric == 'precision':
            return precision_score(actuals, preds)
        elif metric == 'recall':
            return recall_score(actuals, preds)
        elif metric == 'f1_score':
            return f1_score(actuals, preds)
     

    @property
    def test_loss(self):
        return self.model.evaluate(self.window.test, verbose=0)[0]

    def predictions(self, station=None):
        tf_test = self.window.test

        if station is not None:
            filter_index = self.window.stations.index(station)
            num_inputs = len(self.window.train_df.columns.levels[1])
            tf_test = tf_test.unbatch().filter(lambda x, y: tf.math.reduce_sum(x[:, num_inputs + filter_index]) > 0).batch(256)

        return np.squeeze(self.model.predict(tf_test, verbose=0), axis=2)
        
class BaseModel(Model):  
    def __init__(self, model_name, window, conv_width, output_activation='sigmoid', loss_func=tf.losses.MeanSquaredError(), max_epochs=150):
        super().__init__(window)
        
        self.model_name = model_name
        self.mix_type_name = None
        self.loss_func = loss_func
        self.MAX_EPOCHS = max_epochs
        
        if self.model_name == 'multi-linear':          
            self.model = tf.keras.Sequential([
                            # Take the last time step.
                            # Shape [batch, time, features] => [batch, 1, features]
                            tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
                            # Shape => [batch, 1, dense_units]
                            tf.keras.layers.Dense(20, activation='relu'),
                            # Shape => [batch, out_steps*features]
                            tf.keras.layers.Dense(conv_width, activation=output_activation, 
                                                  kernel_initializer=tf.initializers.zeros()),
                            # Shape => [batch, out_steps, features=1]
                            tf.keras.layers.Reshape([conv_width, 1])
                        ])
            
        elif self.model_name == 'multi-CNN':
            self.model = tf.keras.Sequential([
                            # Shape [batch, time, features] => [batch, conv_width, features]
                            tf.keras.layers.Lambda(lambda x: x[:, -conv_width:, :]),
                            # Shape => [batch, 1, conv_units]
                            tf.keras.layers.Conv1D(64, activation='relu', kernel_size=(conv_width)),
                            # Shape => [batch, 1,  out_steps*features]
                            tf.keras.layers.Dense(conv_width, activation=output_activation, 
                                                  kernel_initializer=tf.initializers.zeros()),
                            # Shape => [batch, out_steps, features=1]
                            tf.keras.layers.Reshape([conv_width, 1])
                        ])
            
        elif self.model_name == 'multi-LSTM':                       
            self.model = Sequential([
                            # Shape [batch, time, features] => [batch, lstm_units].
                            # Adding more `lstm_units` just overfits more quickly.
                            LSTM(32, return_sequences=False),
                            # Shape => [batch, out_steps*features].
                            Dense(conv_width, activation=output_activation,
                                                  kernel_initializer=tf.initializers.zeros()),
                            # Shape => [batch, out_steps, features=1].
                            Reshape([conv_width, 1])
                        ])

            
        elif self.model_name == 'multi-ED-LSTM':                       
            self.model = Sequential([
                            # Shape [batch, time, features] => [batch, lstm_units].
                            # Adding more `lstm_units` just overfits more quickly.
                            LSTM(20, return_sequences=True,),
                            # Shape => [batch, out_steps*features].
                            Dropout(0.2),
                            Flatten(),
                            RepeatVector(5),
                            LSTM(20, return_sequences=False), 
                            
                            Dropout(0.2),                
                            Dense(conv_width, activation=output_activation,
                                                  kernel_initializer=tf.initializers.zeros()),
                            # Shape => [batch, out_steps, features=1].
                            Reshape([conv_width, 1])
                        ])    
            
        elif self.model_name == 'multi-Bidirectional-LSTM':                       
            self.model = Sequential([
                            # Shape [batch, time, features] => [batch, lstm_units].
                            # Adding more `lstm_units` just overfits more quickly.
                            Bidirectional(LSTM(20, return_sequences=False)),
                            # Shape => [batch, out_steps*features].
                            Dense(conv_width, activation=output_activation,
                                                  kernel_initializer=tf.initializers.zeros()),
                            # Shape => [batch, out_steps, features=1].
                            Reshape([conv_width, 1])
                        ])  
            
        elif self.model_name == 'multi-deep-Bidirectional-LSTM':                       
            self.model = Sequential([
                            # Shape [batch, time, features] => [batch, lstm_units].
                            # Adding more `lstm_units` just overfits more quickly.
                            Bidirectional(LSTM(64, return_sequences=True
                                              )),
                            Dropout(0.2),
                            Bidirectional(LSTM(32, return_sequences=False)),
                            Dropout(0.2),
                            # Shape => [batch, out_steps*features].
                            Dense(conv_width, activation=output_activation,
                                                  kernel_initializer=tf.initializers.zeros()),
                            # Shape => [batch, out_steps, features=1].
                            Reshape([conv_width, 1])
                        ]) 
            
        self.compile_and_fit(self.model, window, loss_func)



class Mixed_Model(BaseModel):
    threshold = 0.2
    
    def __init__(self, model_name, mix_type_name, window, conv_width):
        super().__init__(model_name, window, conv_width)
        self.mix_type_name = mix_type_name
               
        if self.mix_type_name == 'simple-two_model-onestepAR':
            train_df = self.window.train_df
            test_df = self.window.test_df
            window_simple = WindowGenerator(input_width=1,
                                             label_width=1,
                                             shift=1,
                                             train_df=train_df.loc[:,train_df.columns.get_level_values(1).isin(self.window.label_columns)] ,
                                             test_df=test_df.loc[:,test_df.columns.get_level_values(1).isin(self.window.label_columns)] ,
                                             station=self.window.station,
                                             label_columns=['flood_probabilities'])
            
            self.model_simple = BaseModel(model_name=model_name, window=window_simple, conv_width=1)
            
        elif self.mix_type_name == 'simple-two_model-multistep':
            train_df = self.window.train_df
            test_df = self.window.test_df
            window_simple = WindowGenerator(input_width=1,
                                             label_width=self.window.label_width,
                                             shift=self.window.label_width,
                                             train_df=train_df,
                                             test_df=test_df,
                                             station=self.window.station,
                                             label_columns=['flood_probabilities'])
            
            self.model_simple = BaseModel(model_name=model_name, window=window_simple, conv_width=self.window.label_width)
        elif self.mix_type_name == 'upper_soil-two_model-multistep':
            train_df = self.window.train_df
            test_df = self.window.test_df
            window_simple = WindowGenerator(input_width=self.window.input_width,
                                             label_width=self.window.label_width,
                                             shift=self.window.label_width,
                                             train_df=train_df,
                                             test_df=test_df,
                                             station=self.window.station,
                                            filtered='upper_soil_filter',
                                             label_columns=['flood_probabilities'])
            
            self.model_simple = BaseModel(model_name=model_name, window=window_simple, conv_width=self.window.label_width)
            
            
    @property
    def predictions(self):
        if self.mix_type_name == 'simple':
            preds = super().predictions
            test_array = self.window.test_array[self.window.input_width:]
            new_pred=[]
       
            for pred, actual_before in zip(preds, test_array):
                if actual_before < self.threshold:
                    pred = np.full((self.window.label_width,), actual_before)

                new_pred.append(pred)  
            
            
            
            return np.array(new_pred)
        
        elif self.mix_type_name == 'simple-two_model-onestepAR':
            preds = super().predictions
            preds_simple = self.model_simple.predictions
            
            # test array starts 1 time unit before predictions
            test_array = self.window.test_array[self.window.input_width:]
            
            new_pred=[]

            for pred, actual_before in zip(preds, test_array):
                if actual_before < self.threshold:
                    pred = []
                                      
                    input_value = np.array(actual_before).reshape(1,1,1)
                    
                    for j in range(self.window.label_width):
                        pred_simple = self.model_simple.model.predict(input_value).item()
                        pred.append(pred_simple)
                        
                        input_value = np.array(pred_simple).reshape(1,1,1)
                                         
                    pred = np.array(pred)

                new_pred.append(pred)  
            
            return np.array(new_pred)
        
        elif self.mix_type_name == 'simple-two_model-multistep':
            preds = super().predictions
            preds_simple = self.model_simple.predictions
            
            # test array starts 1 time unit before predictions
            test_array = self.window.test_array[self.window.input_width:]
            
            new_pred=[]

            for i, (pred, actual_before) in enumerate(zip(preds, test_array)):
                if actual_before < self.threshold:                                
                    input_value = self.window.test_example(i+self.window.input_width)
                                              
                    pred = self.model_simple.model.predict(input_value).flatten()

                new_pred.append(pred)  
            
            return np.array(new_pred)
        
        elif self.mix_type_name == 'upper_soil-two_model-multistep':
            preds = super().predictions
            preds_simple = self.model_simple.predictions
            
            # upper soil indicator 1 time unit before predictions
            upper_soil_indicator = self.window.test_indicator(filtered='upper_soil_filter')
                      
            new_pred=[]

            for i, (pred, indicator) in enumerate(zip(preds, upper_soil_indicator)):
                if indicator == 1:                                
                    input_value = self.window.test_example(i+self.window.input_width)
                                              
                    pred = self.model_simple.model.predict(input_value).flatten()

                new_pred.append(pred)              
            
            return np.array(new_pred)  




## Without Changes

class Ensemble_Static():
    epochs = 150
    patience = 5
    def __init__(self, numpy_window, batch_size=256, loss_func=None):
        num_timesteps = numpy_window.input_width
        num_timeseries_features = numpy_window.num_timeseries_features
        num_static_features = numpy_window.num_static_features + numpy_window.total_stations
          
        num_predictions = numpy_window.label_width
        
        self.batch_size = batch_size
        self.stations = numpy_window.stations
        self.n_stations = numpy_window.total_stations
        self.numpy_window = numpy_window

        self.loss_func = loss_func
        # RNN + SLP Model
        # Define input layer

        recurrent_input = Input(shape=(num_timesteps, num_timeseries_features),name="TIMESERIES_INPUT")
        static_input = Input(shape=(num_static_features,),name="STATIC_INPUT")

        # RNN Layers
        # layer - 1
        rec_layer_one = LSTM(64, name ="LSTM_LAYER_1", return_sequences=True)(recurrent_input)
        # rec_layer_one = Dropout(0.1,name ="DROPOUT_LAYER_1")(rec_layer_one)
        
        # layer - 2
        rec_layer_two = LSTM(64, name ="LSTM_LAYER_2", return_sequences=True)(rec_layer_one)
        rec_layer_two = Flatten(name ="FLATTEN_LAYER")(rec_layer_two[:, -2:, :])
        # rec_layer_two = Dropout(0.1,name ="DROPOUT_LAYER_2")(rec_layer_two)
        
        # SLP Layers
        static_layer_one = Dense(64, activation='relu',name="DENSE_LAYER_1")(static_input)
        
        # Combine layers - RNN + SLP
        combined = Concatenate(axis= 1,name = "CONCATENATED_TIMESERIES_STATIC")([rec_layer_two, static_layer_one])
        combined_dense_two = Dense(32, activation='relu',name="DENSE_LAYER_2")(combined)
        output = Dense(num_predictions, activation='sigmoid', name="OUTPUT_LAYER")(combined_dense_two)

        # Compile ModeL
        self.model = keras.models.Model(inputs=[recurrent_input, static_input], outputs=[output])
        # MSE
        
        #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.train_timeseries_x, self.train_static_x, self.train_y = numpy_window.train     
        self.test_timeseries_x, self.test_static_x, self.test_y = numpy_window.test 

        self.train_y = tf.squeeze(self.train_y)
        self.test_y = tf.squeeze(self.test_y)
        
        
        self.model.summary()

        self.train()
        
    def train(self):
        if self.loss_func is not None:
            self.model.compile(loss=self.loss_func, optimizer='adam', metrics=['MeanAbsoluteError'])
        else:
            self.model.compile(loss='MeanSquaredError', optimizer='adam', metrics=['MeanAbsoluteError'])

        # print(self.train_timeseries_x.mean(0).mean(0), self.train_static_x.mean(0), self.train_y.mean(0))
        
        self.model.fit([self.train_timeseries_x, self.train_static_x], 
                       self.train_y, 
                       epochs=self.epochs, 
                       batch_size=self.batch_size, 
                       verbose=0)


    @property
    def test_loss(self):
        return self.model.evaluate([self.test_timeseries_x, self.test_static_x], self.test_y, verbose=0)[0]
    
    def predictions(self, station):  
        filter_index = self.stations.index(station)
        n_observations = int(self.test_static_x.shape[0]/self.n_stations)

        start = int(filter_index*n_observations)
        end = int((filter_index+1)*n_observations)
        
        # print("Timeseries input shape:", self.test_timeseries_x[start:end].shape)
        # print("Static input shape:", self.test_static_x[start:end].shape)
        
        return self.model.predict([self.test_timeseries_x[start:end,:], self.test_static_x[start:end,:]], verbose=0)
    
    
    def actuals(self, station):
        filter_index = self.stations.index(station)
        n_observations = int(self.test_static_x.shape[0]/self.n_stations)

        start = int(filter_index*n_observations)
        end = int((filter_index+1)*n_observations)
        # print('TEST Y SHAPE',self.test_y.shape)

        return self.test_y[start:end, :].numpy()

    
    def average_model_error(self, station, cut=100):
        preds = self.predictions(station)
        actuals = self.actuals(station)
        
        cut_percentile = np.percentile(actuals.flatten(), 100-cut)

        locs = np.where(actuals > cut_percentile)[0]
        preds = preds[locs]
        actuals = actuals[locs]

        avg_error = 0

        for window_pred, window_actual in zip(preds, actuals):
            avg_error += np.sum((window_pred - window_actual)**2)
        
        if avg_error==0:
            return 0
        avg_error = avg_error/(actuals.shape[0]*actuals.shape[1])

        return avg_error 
    
    def print_model_windows(self, station, cut=100):
        preds = self.predictions(station)
        actuals = self.actuals(station)
        cut_percentile = np.percentile(actuals.flatten(), 100-cut)

        locs = np.where(actuals > cut_percentile)[0]
        preds = preds[locs]
        actuals = actuals[locs]
        
        for pred, actual, loc in zip(preds, actuals, locs):
            print("time: {}".format(loc))
            print("Input: {}".format(self.test_y[loc:loc+self.numpy_window.input_width+1].flatten()))
            print("Predicted: {}".format(pred))
            print("Actual: {}".format(actual))
            print("-------------------------")

    def summary(self, station):
        summary_dict = {}
        
        summary_dict['station'] = station
        summary_dict['input_width'] = self.numpy_window.input_width
        summary_dict['label_width'] = self.numpy_window.label_width
        summary_dict['num_timeseries_features'] = self.numpy_window.num_timeseries_features
        summary_dict['num_static_features'] = self.numpy_window.num_static_features
        summary_dict['timeseries_inputs'] = self.numpy_window.timeseries_source
        summary_dict['static_inputs'] = self.numpy_window.summary_source

   
        summary_dict['SER_1%'] = self.average_model_error(station, cut=1)
        summary_dict['SER_2%'] = self.average_model_error(station, cut=2)
        summary_dict['SER_5%'] = self.average_model_error(station, cut=5)
        summary_dict['SER_10%'] = self.average_model_error(station, cut=10)
        summary_dict['SER_25%'] = self.average_model_error(station, cut=25)  
        summary_dict['SER_50%'] = self.average_model_error(station, cut=50)  
        summary_dict['SER_75%'] = self.average_model_error(station, cut=75)  
        summary_dict['MSE'] = self.average_model_error(station, cut=100)
        
        return summary_dict





class Switch_Model(Model):
    threshold = 0.7
    
    def __init__(self, window_switch, window_regular, conv_width):
        self.window_switch = window_switch
        self.window = window_regular
        
        assert(self.window_switch.input_width == self.window.input_width)
        
        self.switch = Ensemble_Static(window_switch)
        
        self.regular = BaseModel(model_name='multi-LSTM', window=window_regular, conv_width=conv_width)
        self.q70 = BaseModel(model_name='multi-LSTM', window=window_regular, conv_width=conv_width, loss_func=QuantileLoss.qloss_70)
        self.q95 = BaseModel(model_name='multi-LSTM', window=window_regular, conv_width=conv_width, loss_func=QuantileLoss.qloss_95)
        
    def predictions(self, station):
        preds_switch = self.switch.predictions(station)
        
        preds_regular = self.regular.predictions(station)
        preds_q70 = self.q70.predictions(station)
        preds_q95 = self.q95.predictions(station)

        test_array = self.window.test_windows(station)

        new_pred = []
        
        # for pred_switch, pred_regular, pred_q70, pred_q95 in zip(preds_switch, preds_regular, preds_q70, preds_q95):

        #     switch_condition = pred_switch >= 0.95
        #     q95_condition = (pred_switch >= 0.7) & (pred_switch < 0.95)
        #     q70_condition = pred_switch < 0.7  # You might want to specify this condition differently

        #     new_pred.append(np.where(switch_condition, pred_q95, np.where(q95_condition, pred_q70, pred_regular)))

        switch_condition = preds_switch >= 0.95
        q95_condition = (preds_switch >= 0.7) & (preds_switch < 0.95)
        q70_condition = preds_switch < 0.7  # You might want to specify this condition differently

        new_pred = np.where(switch_condition, preds_q95, np.where(q95_condition, preds_q70, preds_regular))
        
        return new_pred
        

    def test_MSE(self, station=None):
        preds = self.predictions(data='test', station=station)
        test_array = self.window.test_array(station)[self.window.input_width:]

        return mean_squared_error(test_array, preds)
    
    def test_ROCAUC(self, station, level=0.05):
        preds = self.predictions(data='test', station=station)
        test_array = (self.window.test_array(station)[self.window.input_width:] < level).astype(int)
        
        return roc_auc_score(test_array, preds)


    def average_model_error_between_quantiles(self, station=None, q_start=None, q_end=None):
            
            
        if station is not None:
            preds_all = self.predictions(station)
            actuals_all = self.window.test_windows(station)
        else:
            preds_all = self.predictions()
            actuals_all = self.window.test_windows()

        print(f"station: {station} -- q_start: {q_start}, q_end: {q_end}")

        avg_error_all = []

        for t in range(actuals_all.shape[-1]):

            preds = preds_all[...,t]
            actuals = actuals_all[...,t]

            if q_start is not None and q_end is not None:
                start_percentile = np.percentile(actuals.flatten(), q_start)
                end_percentile = np.percentile(actuals.flatten(), q_end)
                locs = np.where((actuals>=start_percentile) & (actuals<end_percentile))[0]
            
            elif q_start is not None and q_end is None:
                start_percentile = np.percentile(actuals.flatten(), q_start)
                end_percentile = None
                locs = np.where(actuals>=start_percentile)[0]
    
            elif q_start is None and q_end is not None:
                start_percentile = None
                end_percentile = np.percentile(actuals.flatten(), q_end)
                locs = np.where(actuals<end_percentile)[0]
    
            else:
                raise(Exception('Both q_start and q_end cannot be None'))
    
    
            print(f"Station: {station} -- Quantiles: {start_percentile} -- {end_percentile}")
    
            if len(locs) != 0:
                preds = preds[locs]
                actuals = actuals[locs]
                
                avg_error = 0
        
                for window_pred, window_actual in zip(preds, actuals):
                    avg_error += np.sum((window_pred - window_actual)**2)
        
                avg_error = avg_error/actuals.shape[0]
            
            else:
                avg_error = np.nan

            avg_error_all.append(avg_error)


        return avg_error_all

    

    def summary(self, station=None, **kwargs):
        summary_dict = {}
        
        summary_dict['input_width'] = self.window.input_width
        summary_dict['label_width'] = self.window.label_width

        if station is not None:
            summary_dict['station'] = station

        summary_dict['NSE'] = self.get_NSE(station, **kwargs)
                  
        # summary_dict['SER_1%'] = self.average_model_error(station, cut=1)
        # summary_dict['SER_2%'] = self.average_model_error(station, cut=2)
        # summary_dict['SER_5%'] = self.average_model_error(station, cut=5)
        # summary_dict['SER_10%'] = self.average_model_error(station, cut=10)
        # summary_dict['SER_25%'] = self.average_model_error(station, cut=25)
        # summary_dict['SER_50%'] = self.average_model_error(station, cut=50)
        # summary_dict['SER_75%'] = self.average_model_error(station, cut=75)
        # summary_dict['RMSE'] = self.average_model_error(station, cut=100)

        summary_dict['MSE_high'] = self.average_model_error_between_quantiles(station, q_start=70, q_end=95)
        summary_dict['MSE_extreme'] = self.average_model_error_between_quantiles(station, q_start=95)
        summary_dict['MSE_normal'] = self.average_model_error_between_quantiles(station, q_end=70)
        summary_dict['MSE_all'] = self.average_model_error_between_quantiles(station, q_end=100)

        return summary_dict


class QuantileEnsemble(Model):
    
    threshold = 0.7
    
    def __init__(self, window, conv_width, max_epochs=250):
        self.window = window
        self.regular = BaseModel(model_name='multi-LSTM',
                                  window=window,
                                  conv_width=conv_width,
                                  max_epochs=max_epochs)
        self.q05 = BaseModel(model_name='multi-LSTM',
                             window=window,
                             conv_width=conv_width,
                             loss_func=QuantileLoss.qloss_05,
                             max_epochs=max_epochs)
        self.q95 = BaseModel(model_name='multi-LSTM',
                             window=window,
                             conv_width=conv_width,
                             loss_func=QuantileLoss.qloss_95,
                             max_epochs=max_epochs)
        
    def predictions(self, station):
        preds_regular = self.regular.predictions(station)
        preds_q05 = self.q05.predictions(station)
        preds_q95 = self.q95.predictions(station)
        return np.stack([preds_regular, preds_q05, preds_q95], axis=-1)

    def test_MSE(self, station=None):
        preds = self.predictions(data='test', station=station)
        test_array = self.window.test_array(station)[self.window.input_width:]
        return mean_squared_error(test_array, preds)
    
    def interval_score(self, target, pred_low, pred_high, alpha=0.1):
        # Calculate the interval score
        L = pred_low
        U = pred_high
        IS = (U - L) + (2/alpha) * np.maximum(0, L - target) + (2/alpha) * np.maximum(0, target - U)
        return IS

    def summary(self, station=None, **kwargs):

        summary_regular = self.regular.summary(station, **kwargs)
        summary_q05 = self.q05.summary(station, **kwargs)
        summary_q95 = self.q95.summary(station, **kwargs)

        if kwargs.get('conf_score', False):
            predictions = self.predictions(station)
            actuals = self.window.test_windows(station)

            # min and scale values for the station
            _min = kwargs.get('_min', None)
            _scale = kwargs.get('_scale', None)

            preds_q05 = predictions[...,1]
            preds_q95 = predictions[...,2]

            if _min is not None and _scale is not None:
                preds_q05 = (preds_q05 - _min)/_scale
                preds_q95 = (preds_q95 - _min)/_scale
                actuals = (actuals - _min)/_scale

            conf = []
            for i in range(self.window.label_width):
                # window_flag = (actuals[:, i] > preds_q05[:, i]) & (actuals[:, i] < preds_q95[:, i])
                # conf.append(np.abs(window_flag.sum()/window_flag.shape[0] - 0.90))
                conf.append(self.interval_score(target=actuals[:, i],
                                                pred_low=preds_q05[:, i], 
                                                pred_high=preds_q95[:, i], 
                                                alpha=0.1))
            
            conf = np.mean(conf)

            return [summary_regular, summary_q05, summary_q95, conf]
        
        return [summary_regular, summary_q05, summary_q95]