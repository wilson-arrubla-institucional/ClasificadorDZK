# Importar librerías
import streamlit as st
import pickle
import pandas as pd
import base64
import sklearn

# Configuración de la página (debe ser la primera instrucción)
############################################################################################################################

st.set_page_config(page_title="Clasificador diferencial del Dengue, Zika y Chikungunya", layout="centered")
    # Título principal centrado
# Función para cargar imágenes locales como base64
def load_image_as_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Convertir imagen a base64
image_base64 = load_image_as_base64("LogoApp.png")

# HTML con la imagen convertida
st.markdown(
    f"""
    <style>
    .top-right {{
        position: absolute;
        top: -70px;
        right: -300px; /* Reduce este valor para acercar la imagen más al borde derecho */
    }}
    </style>
    <div class="top-right">
        <img src="data:image/png;base64,{image_base64}" alt="Logo" width="350">
    </div>
    """,
    unsafe_allow_html=True
)



############################################################################################################################

# Cargar el Pre-modelo para obtener el target en la transformación de los datos
filename = 'DZC-modelTree.pkl'
modelTree, labelencoder, variables = pickle.load(open(filename, 'rb'))

# Cargar el modelo para trabajar con la metodologia basada en la evidencia científica 
filename_1 = 'modelo-clas-tree.pkl'
mod_Tree, label_encoder, variables_1 = pickle.load(open(filename_1, 'rb'))

def main():
    # Título principal
    st.title("Clasificador Diferencial del Dengue, Zika y Chikungunya")
    #Titulo de Sidebar
    st.sidebar.header('Signos y sintomas del paciente')
# Entradas del usuario

 # Entradas del usuario en el Sidebar

    def user_input_features():
        # Edad como valor entero , entre 0 y 60 años)
        edad = st.sidebar.slider('Edad', min_value=1, max_value=100, value=25, step=1)  # step=1 garantiza que se seleccionen valores enteros
        dia_sint = st.sidebar.slider('Día sintomas', min_value=1, max_value=12, value=6, step=1)  # step=1 garantiza que se seleccionen valores enteros
        
        # Seleccionar signos y sintomas
        option = ['No', 'Yes']
        option_sex = ['F', 'M']

        Sexo = st.sidebar.selectbox('Sex', option_sex)
        headache = st.sidebar.selectbox('headache', option)
        Retroocular_pain = st.sidebar.selectbox('Dolor Retroocular', option)
        Myalgia = st.sidebar.selectbox('Mialgia', option)
        Arthralgia = st.sidebar.selectbox('Artralgia', option)
        Rash = st.sidebar.selectbox('Erupción cutánea', option)
        Abdominal_pain = st.sidebar.selectbox('Dolor Abdominal', option)
        Threw_up = st.sidebar.selectbox('Vómito', option)
        Diarrhea = st.sidebar.selectbox('Diarrea', option)
        Drowsiness = st.sidebar.selectbox('Somnolencia o Mareos', option)
        Hepatomegaly = st.sidebar.selectbox('Hepatomegalia', option)
        Mucosal_hemorrhage = st.sidebar.selectbox('Hemorragias en las mucosas', option)
        Hyperemia = st.sidebar.selectbox('Hiperemia', option)
        exanthema = st.sidebar.selectbox('Exantema', option)          
    
        data = {
            'Age': edad,
            'Symptom_days': dia_sint,
            'Sex': Sexo,
            'headache': headache,
            'Retroocular_pain': Retroocular_pain,
            'Myalgia': Myalgia,
            'Arthralgia': Arthralgia,
            'Rash': Rash,
            'Abdominal_pain': Abdominal_pain,
            'Threw_up': Threw_up,
            'Diarrhea': Diarrhea,
            'Drowsiness': Drowsiness,
            'Hepatomegaly': Hepatomegaly,
            'Mucosal_hemorrhage': Mucosal_hemorrhage,
            'Hyperemia': Hyperemia,
            'exanthema': exanthema
        }
        features = pd.DataFrame(data, index=[0])
        return features
    
    # Preparar los datos
    df = user_input_features()  # permite ver en el front el sidebar
        
# Creamos método preparar datos 

    def prepara_datos():
        data_preparada = df.copy()
        data_preparada = pd.get_dummies(data_preparada, columns=['Sex', 'headache', 'Retroocular_pain', 'Myalgia',
                                                        'Arthralgia', 'Rash', 'Abdominal_pain', 'Threw_up',
                                                        'Diarrhea', 'Drowsiness', 'Hepatomegaly', 'Mucosal_hemorrhage',
                                                        'Hyperemia', 'exanthema'], drop_first=False)
            
    # Se añaden las columnas faltantes en este caso si falta alguna dummy, se creará y se llenará con ceros
        data_preparada = data_preparada.reindex(columns=variables, fill_value=0)
   
        return data_preparada
    
        
    # Llamada a la función
   
    df_preparados = prepara_datos()

    st.subheader('Valores seleccionados')
    st.write(df_preparados)

    # Crear un botón para realizar la predicción
    if st.button('Realizar Predicción'):
        Y_fut = modelTree.predict(df_preparados)
        resultado = labelencoder.inverse_transform(Y_fut)
        #st.success(f'La predicción es: {(resultado[0])}')
#########################################################################################################################
#Preparación para modelo con pesos de la evidencia cientifica#

        def asignacion_pesos_evidencia():
            datos1 = df.copy()
            datos1['Target'] = resultado[0]
            #st.write(datos1)
           #st.write(f'predicción: {(resultado[0])}')#Revisar el dataframe con la predicción Target
        # Definición  rangos específicos certeza de la evidencia
            Muy_bajo_value_0 =0.0
            Muy_bajo_value_1 =0.25
            Bajo_value_0 = 0.26
            Bajo_value_1 = 0.50
            Moderado_value_0 = 0.51
            Moderado_value_1 = 0.75
            Alto_value_0 = 0.76
            Alto_value_1 =1

            #Limites de los días del transcurso de las enfermedades
            x_min = 0
            x_max = 12

            # Función para interpolación lineal entre dos valores

            def interpolate(min_value, max_value, x, x_min, x_max):
                return min_value + (max_value - min_value) * ((x - x_min) / (x_max - x_min))

            # Asignación basada en interpolación lineal dentro del rango especificado
            #Mialgia
            for index, row in datos1.iterrows():
                if row['Myalgia'] == "Yes" and row['Target'] == "Chikungunya":
                    datos1.at[index, 'Myalgia'] = interpolate(Moderado_value_0, Moderado_value_1, row['Symptom_days'], x_min, x_max)
                elif row['Myalgia'] == "Yes" and row['Target'] == "Dengue":
                    datos1.at[index, 'Myalgia'] = interpolate(Muy_bajo_value_0, Muy_bajo_value_1, row['Symptom_days'], x_min, x_max)
                elif row['Myalgia'] == "Yes" and row['Target'] == "Zika":
                    datos1.at[index, 'Myalgia'] = interpolate(Muy_bajo_value_0, Muy_bajo_value_1, row['Symptom_days'], x_min, x_max)

                elif row['Myalgia'] == "No" and row['Target'] == "Chikungunya":
                    datos1.at[index, 'Myalgia'] = 0
                elif row['Myalgia'] == "No" and row['Target'] == "Dengue":
                    datos1.at[index, 'Myalgia'] = 0
                elif row['Myalgia'] == "No" and row['Target'] == "Zika":
                    datos1.at[index, 'Myalgia'] = 0
            datos1['Myalgia'] = pd.to_numeric(datos1['Myalgia'], errors='coerce')

            # Cefalea
            for index, row in datos1.iterrows():
                if row['headache'] == "Yes" and row['Target'] == "Chikungunya":
                    datos1.at[index, 'headache'] = interpolate(Muy_bajo_value_0, Muy_bajo_value_1, row['Symptom_days'], x_min, x_max)
                elif row['headache'] == "Yes" and row['Target'] == "Dengue":
                    datos1.at[index, 'headache'] = interpolate(Bajo_value_0, Bajo_value_1, row['Symptom_days'], x_min, x_max)
                elif row['headache'] == "Yes" and row['Target'] == "Zika":
                    datos1.at[index, 'headache'] = interpolate(Muy_bajo_value_0, Muy_bajo_value_1, row['Symptom_days'], x_min, x_max)


                elif row['headache'] == "No" and row['Target'] == "Chikungunya":
                    datos1.at[index, 'headache'] = 0
                elif row['headache'] == "No" and row['Target'] == "Dengue":
                    datos1.at[index, 'headache'] = 0
                elif row['headache'] == "No" and row['Target'] == "Zika":
                    datos1.at[index, 'headache'] = 0
            datos1['headache'] = pd.to_numeric(datos1['headache'], errors='coerce')

        #Dolor Retroocular
            for index, row in datos1.iterrows():
                if row['Retroocular_pain'] == "Yes" and row['Target'] == "Chikungunya":
                    datos1.at[index, 'Retroocular_pain'] = interpolate(Muy_bajo_value_0, Muy_bajo_value_1, row['Symptom_days'], x_min, x_max)
                elif row['Retroocular_pain'] == "Yes" and row['Target'] == "Dengue":
                    datos1.at[index, 'Retroocular_pain'] = interpolate(Bajo_value_0, Bajo_value_1, row['Symptom_days'], x_min, x_max)
                elif row['Retroocular_pain'] == "Yes" and row['Target'] == "Zika":
                    datos1.at[index, 'Retroocular_pain'] = interpolate(Muy_bajo_value_0, Muy_bajo_value_1, row['Symptom_days'], x_min, x_max)


                elif row['Retroocular_pain'] == "No" and row['Target'] == "Chikungunya":
                    datos1.at[index, 'Retroocular_pain'] = 0
                elif row['Retroocular_pain'] == "No" and row['Target'] == "Dengue":
                    datos1.at[index, 'Retroocular_pain'] = 0
                elif row['Retroocular_pain'] == "No" and row['Target'] == "Zika":
                    datos1.at[index, 'Retroocular_pain'] = 0
                
            datos1['Retroocular_pain'] = pd.to_numeric(datos1['Retroocular_pain'], errors='coerce')

        # Artralgia
            for index, row in datos1.iterrows():
                if row['Arthralgia'] == "Yes" and row['Target'] == "Chikungunya":
                    datos1.at[index, 'Arthralgia'] = interpolate(Alto_value_0, Alto_value_1, row['Symptom_days'], x_min, x_max)
                elif row['Arthralgia'] == "Yes" and row['Target'] == "Dengue":
                    datos1.at[index, 'Arthralgia'] = interpolate(Muy_bajo_value_0, Muy_bajo_value_1, row['Symptom_days'], x_min, x_max)
                elif row['Arthralgia'] == "Yes" and row['Target'] == "Zika":
                    datos1.at[index, 'Arthralgia'] = interpolate(Muy_bajo_value_0, Muy_bajo_value_1, row['Symptom_days'], x_min, x_max)

                elif row['Arthralgia'] == "No" and row['Target'] == "Chikungunya":
                    datos1.at[index, 'Arthralgia'] = 0
                elif row['Arthralgia'] == "No" and row['Target'] == "Dengue":
                    datos1.at[index, 'Arthralgia'] = 0
                elif row['Arthralgia'] == "No" and row['Target'] == "Zika":
                    datos1.at[index, 'Arthralgia'] = 0
            datos1['Arthralgia'] = pd.to_numeric(datos1['Arthralgia'], errors='coerce')


        #Erupción Rash
            for index, row in datos1.iterrows():
                if row['Rash'] == "Yes" and row['Target'] == "Chikungunya":
                    datos1.at[index, 'Rash'] = interpolate(Moderado_value_0, Moderado_value_1, row['Symptom_days'], x_min, x_max)
                elif row['Rash'] == "Yes" and row['Target'] == "Dengue":
                    datos1.at[index, 'Rash'] = interpolate(Muy_bajo_value_0, Muy_bajo_value_1, row['Symptom_days'], x_min, x_max)
                elif row['Rash'] == "Yes" and row['Target'] == "Zika":
                    datos1.at[index, 'Rash'] = interpolate(Moderado_value_0, Moderado_value_1, row['Symptom_days'], x_min, x_max)

                elif row['Rash'] == "No" and row['Target'] == "Chikungunya":
                    datos1.at[index, 'Rash'] = 0
                elif row['Rash'] == "No" and row['Target'] == "Dengue":
                    datos1.at[index, 'Rash'] = 0
                elif row['Rash'] == "No" and row['Target'] == "Zika":
                    datos1.at[index, 'Rash'] = 0
            datos1['Rash'] = pd.to_numeric(datos1['Rash'], errors='coerce')

        #Dolor Abdominal
            for index, row in datos1.iterrows():
                if row['Abdominal_pain'] == "Yes" and row['Target'] == "Dengue":
                    datos1.at[index, 'Abdominal_pain'] = interpolate(Moderado_value_0, Moderado_value_1, row['Symptom_days'], x_min, x_max)
                elif row['Abdominal_pain'] == "Yes" and row['Target'] == "Chikungunya":
                    datos1.at[index, 'Abdominal_pain'] = interpolate(Muy_bajo_value_0, Muy_bajo_value_1, row['Symptom_days'], x_min, x_max)
                elif row['Abdominal_pain'] == "Yes" and row['Target'] == "Zika":
                    datos1.at[index, 'Abdominal_pain'] = interpolate(Muy_bajo_value_0, Muy_bajo_value_1, row['Symptom_days'], x_min, x_max)

                elif row['Abdominal_pain'] == "No" and row['Target'] == "Chikungunya":
                    datos1.at[index, 'Abdominal_pain'] = 0
                elif row['Abdominal_pain'] == "No" and row['Target'] == "Dengue":
                    datos1.at[index, 'Abdominal_pain'] = 0
                elif row['Abdominal_pain'] == "No" and row['Target'] == "Zika":
                    datos1.at[index, 'Abdominal_pain'] = 0
            datos1['Abdominal_pain'] = pd.to_numeric(datos1['Abdominal_pain'], errors='coerce')
        
        #Vómitos
            for index, row in datos1.iterrows():
                if row['Threw_up'] == "Yes" and row['Target'] == "Dengue":
                    datos1.at[index, 'Threw_up'] = interpolate(Moderado_value_0, Moderado_value_1, row['Symptom_days'], x_min, x_max)
                elif row['Threw_up'] == "Yes" and row['Target'] == "Chikungunya":
                    datos1.at[index, 'Threw_up'] = interpolate(Muy_bajo_value_0, Muy_bajo_value_1, row['Symptom_days'], x_min, x_max)
                elif row['Threw_up'] == "Yes" and row['Target'] == "Zika":
                    datos1.at[index, 'Threw_up'] = interpolate(Muy_bajo_value_0, Muy_bajo_value_1, row['Symptom_days'], x_min, x_max)

                elif row['Threw_up'] == "No" and row['Target'] == "Chikungunya":
                    datos1.at[index, 'Threw_up'] = 0
                elif row['Threw_up'] == "No" and row['Target'] == "Dengue":
                    datos1.at[index, 'Threw_up'] = 0
                elif row['Threw_up'] == "No" and row['Target'] == "Zika":
                    datos1.at[index, 'Threw_up'] = 0
            datos1['Threw_up'] = pd.to_numeric(datos1['Threw_up'], errors='coerce')

        # Diarrea
            for index, row in datos1.iterrows():
                if row['Diarrhea'] == "Yes" and row['Target'] == "Dengue":
                    datos1.at[index, 'Diarrhea'] = interpolate(Bajo_value_0, Bajo_value_1, row['Symptom_days'], x_min, x_max)
                elif row['Diarrhea'] == "Yes" and row['Target'] == "Chikungunya":
                    datos1.at[index, 'Diarrhea'] = interpolate(Muy_bajo_value_0, Muy_bajo_value_1, row['Symptom_days'], x_min, x_max)
                elif row['Diarrhea'] == "Yes" and row['Target'] == "Zika":
                    datos1.at[index, 'Diarrhea'] = interpolate(Muy_bajo_value_0, Muy_bajo_value_1, row['Symptom_days'], x_min, x_max)

                elif row['Diarrhea'] == "No" and row['Target'] == "Chikungunya":
                    datos1.at[index, 'Diarrhea'] = 0
                elif row['Diarrhea'] == "No" and row['Target'] == "Dengue":
                    datos1.at[index, 'Diarrhea'] = 0
                elif row['Diarrhea'] == "No" and row['Target'] == "Zika":
                    datos1.at[index, 'Diarrhea'] = 0
            datos1['Diarrhea'] = pd.to_numeric(datos1['Diarrhea'], errors='coerce')
        
        #Hepatomegalia
            for index, row in datos1.iterrows():
                if row['Hepatomegaly'] == "Yes" and row['Target'] == "Dengue":
                    datos1.at[index, 'Hepatomegaly'] = interpolate(Bajo_value_0, Bajo_value_1, row['Symptom_days'], x_min, x_max)
                elif row['Hepatomegaly'] == "Yes" and row['Target'] == "Chikungunya":
                    datos1.at[index, 'Hepatomegaly'] = interpolate(Muy_bajo_value_0, Muy_bajo_value_1, row['Symptom_days'], x_min, x_max)
                elif row['Hepatomegaly'] == "Yes" and row['Target'] == "Zika":
                    datos1.at[index, 'Hepatomegaly'] = interpolate(Muy_bajo_value_0, Muy_bajo_value_1, row['Symptom_days'], x_min, x_max)

                elif row['Hepatomegaly'] == "No" and row['Target'] == "Chikungunya":
                    datos1.at[index, 'Hepatomegaly'] = 0
                elif row['Hepatomegaly'] == "No" and row['Target'] == "Dengue":
                    datos1.at[index, 'Hepatomegaly'] = 0
                elif row['Hepatomegaly'] == "No" and row['Target'] == "Zika":
                    datos1.at[index, 'Hepatomegaly'] = 0
            datos1['Hepatomegaly'] = pd.to_numeric(datos1['Hepatomegaly'], errors='coerce')

        #HEMORRAGIAS EN MUCOSAS
            for index, row in datos1.iterrows():
                if row['Mucosal_hemorrhage'] == "Yes" and row['Target'] == "Dengue":
                    datos1.at[index, 'Mucosal_hemorrhage'] = interpolate(Moderado_value_0, Moderado_value_1, row['Symptom_days'], x_min, x_max)
                elif row['Mucosal_hemorrhage'] == "Yes" and row['Target'] == "Chikungunya":
                    datos1.at[index, 'Mucosal_hemorrhage'] = interpolate(Bajo_value_0, Bajo_value_1, row['Symptom_days'], x_min, x_max)
                elif row['Mucosal_hemorrhage'] == "Yes" and row['Target'] == "Zika":
                    datos1.at[index, 'Mucosal_hemorrhage'] = interpolate(Muy_bajo_value_0, Muy_bajo_value_1, row['Symptom_days'], x_min, x_max)

                elif row['Mucosal_hemorrhage'] == "No" and row['Target'] == "Chikungunya":
                    datos1.at[index, 'Mucosal_hemorrhage'] = 0
                elif row['Mucosal_hemorrhage'] == "No" and row['Target'] == "Dengue":
                    datos1.at[index, 'Mucosal_hemorrhage'] = 0
                elif row['Mucosal_hemorrhage'] == "No" and row['Target'] == "Zika":
                    datos1.at[index, 'Mucosal_hemorrhage'] = 0
            datos1['Mucosal_hemorrhage'] = pd.to_numeric(datos1['Mucosal_hemorrhage'], errors='coerce')


            return datos1
        # Llamada a la función
    
        df_datos_pesos = asignacion_pesos_evidencia()
        # Eliminar la columna "Target" y reasignar el DataFrame
        df_datos_pesos = df_datos_pesos.drop("Target", axis=1)
        # Metodo para preparar datos basados en la evidencia cientifica
        #st.write(df_datos_pesos)

        def prepara_datos_evidencia():
                # Copiar los datos originales
            dato_prepa = df_datos_pesos.copy()
            
            # Crear variables dummies para las columnas categóricas
            dummy_columns = ['Sex', 'Drowsiness', 'Hyperemia', 'exanthema']
            dato_prepa = pd.get_dummies(dato_prepa, columns=dummy_columns, drop_first=False)
            
            # Lista de columnas esperadas en el modelo (en el orden correcto)
            expected_columns = [
                'Age', 'Symptom_days', 'headache', 'Retroocular_pain', 'Myalgia', 'Arthralgia', 
                'Rash', 'Abdominal_pain', 'Threw_up', 'Diarrhea', 'Hepatomegaly', 
                'Mucosal_hemorrhage', 'Sex_M', 'Drowsiness_Yes', 'Hyperemia_Yes', 'exanthema_Yes'
            ]
            
            # Asegurarse de que todas las columnas esperadas estén presentes en el DataFrame
            for col in expected_columns:
                if col not in dato_prepa.columns:
                    # Si una columna está ausente, se crea con valores por defecto (0 en este caso)
                    dato_prepa[col] = 0
            
            # Ordenar las columnas según el orden esperado
            dato_prepa = dato_prepa[expected_columns]
            
            return dato_prepa
                                

                

        data = prepara_datos_evidencia() 
        #Revisar dataframe antes de pasarlo por el modelo
        #st.subheader('Valores ')
        #st.write(data)
        #st.write(data.dtypes)

        Y_fut_2 = mod_Tree.predict(data)
        resultado_2 = label_encoder.inverse_transform(Y_fut_2)
        st.success(f'La predicción es: {(resultado_2[0])}')

    
    # Crear un botón para limpiar por defecto
        if st.button('Limpiar'):
            user_input_features()
            st.session_state.clear()
            st.experimental_rerun()

if __name__ == '__main__':
    main()