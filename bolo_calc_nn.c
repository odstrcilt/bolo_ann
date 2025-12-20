#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "./include/k2c_tensor_include.h"
#include "boloNN.h"

static struct k2c_tensor k2c_buildTensor(float * array, size_t sz) {
	return (struct k2c_tensor){ array, 1, sz, {sz} };
}

static void bolo_calc_nn(
    float const input_bolometry[44],  // Bolometry signals
    float const efit_data[11],      // EFIT data
    float * p_sol, float * p_ldivL, float * p_ldivR, float * p_udivL, float * p_udivR,
    float * p_ldiv, float * p_udiv, float * p_core, float * p_axis, float * p_tot)
{


    // Bolometry signal normalizations
    static float const bolo_normalizations[44] = {
        2.9034e8f, 2.8066e8f, 2.7273e8f, 2.6635e8f, 4.0340e8f, 3.9855e8f, 3.9488e8f, 3.9235e8f,
        3.9091e8f, 3.9055e8f, 3.9126e8f, 0.7972e8f, 0.817e8f,  0.8498e8f, 0.7549e8f, 0.7129e8f,
        0.6854e8f, 1.1162e8f, 1.107e8f,  1.1081e8f, 1.1196e8f, 1.1419e8f, 1.1761e8f, 2.9321e8f,
        2.8825e8f, 2.8449e8f, 2.8187e8f, 2.8033e8f, 0.7058e8f, 0.7140e8f, 0.7334e8f, 0.7657e8f,
        0.8136e8f, 0.7112e8f, 0.6654e8f, 0.6330e8f, 0.6123e8f, 2.9621e8f, 2.9485e8f, 2.9431e8f,
        2.9756e8f, 3.0032e8f, 3.0397e8f, 0.6406e8f
    };

    float processed_bolo[44];
    float processed_efit[11];
    
    // Normalize bolometry
    for (size_t i = 0; i < 44; ++i) {
        processed_bolo[i] = input_bolometry[i] * bolo_normalizations[i];
    }
    
    // Copy EFIT equilibrium data
    for (size_t i = 0; i < 11; ++i) processed_efit[i] = efit_data[i];
    
    // Apply EFIT validity constraints
    if ((processed_efit[10] > 0.3f) || (processed_efit[10] < -0.3f)) processed_efit[10] = 0.0f;
    if ((processed_efit[0] > 1.8f) || (processed_efit[0] < 0.0f)) {
        processed_efit[2] = -1.15f; processed_efit[0] = 1.25f;
    }
    if ((processed_efit[1] > 1.8f) || (processed_efit[1] < 0.0f)) {
        processed_efit[1] = 1.2f; processed_efit[3] = 1.2f;
    }
    
    float nn_output_flat[440];
    
    // Build input/output tensors and apply neural network
    k2c_tensor input_tensor = k2c_buildTensor(processed_efit, sizeof(processed_efit) / sizeof(processed_efit[0]));
    k2c_tensor output_tensor = k2c_buildTensor(nn_output_flat, sizeof(nn_output_flat) / sizeof(nn_output_flat[0]));
    boloNN(&input_tensor, &output_tensor);
    
    // Debug: print first five network output values
    printf("\nNN output (first 5 values):\n");
    for (size_t i = 0; i < 5 && i < output_tensor.numel; ++i) {
        printf("  nn_output[%zu] = %f\n", i, nn_output_flat[i]);
    }

    // Compute power outputs: sum normalized bolometry × network weights
    float p_out[10] = {0.0f};
    for (size_t p = 0; p < 10; ++p) {
        for (size_t c = 0; c < 44; ++c) {
            p_out[p] += processed_bolo[c] * nn_output_flat[p * 44 + c];
        }
    }
    
    // Assign power outputs
    *p_sol = p_out[0];    *p_ldivL = p_out[1];  *p_ldivR = p_out[2];
    *p_udivL = p_out[3];  *p_udivR = p_out[4];  *p_ldiv = p_out[5];
    *p_udiv = p_out[6];   *p_core = p_out[7];   *p_axis = p_out[8];
    *p_tot = p_out[9];
}

int main() {
    printf("=== Starting bolo_calc_nn Test ===\n\n");

    // Setup test inputs: 44-channel bolometry and 11-element EFIT data
    float raw_bolo[44];
    for (size_t i = 0; i < 44; ++i) raw_bolo[i] = 1.0f;

    // EFIT equilibrium data
    float efit[11] = { 
        1.25f, 1.20f, -1.15f, 1.20f, 0.0f, 0.0f, 0.0f, 0.0f, 1.8f, 0.6f, 0.01f
    };

    // Output power channels
    float p_sol, p_ldivL, p_ldivR, p_udivL, p_udivR, p_ldiv, p_udiv, p_core, p_axis, p_tot;

    // Run inference 5 times and display results
    printf("Running inference for 5 iterations...\n");
    for(int i=0; i<5; i++) {
        bolo_calc_nn(raw_bolo, efit,
            &p_sol, &p_ldivL, &p_ldivR, &p_udivL, &p_udivR,
            &p_ldiv, &p_udiv, &p_core, &p_axis, &p_tot
        );
        printf("Step %d | P_sol: %10.2e\n", i+1, p_sol);
    }

    // Display final power outputs from neural network
    printf("\n=== Power Outputs ===\n");
    printf("P_sol:   %.4e\n", p_sol);
    printf("P_ldivL: %.4e\n", p_ldivL);
    printf("P_ldivR: %.4e\n", p_ldivR);
    printf("P_udivL: %.4e\n", p_udivL);
    printf("P_udivR: %.4e\n", p_udivR);
    printf("P_ldiv:  %.4e\n", p_ldiv);
    printf("P_udiv:  %.4e\n", p_udiv);
    printf("P_core:  %.4e\n", p_core);
    printf("P_axis:  %.4e\n", p_axis);
    printf("P_tot:   %.4e\n", p_tot);


    return 0;
}