`timescale 1ns/1ps

// Sequential fully connected engine.
// score[class] = bias[class] + sum(feature[i] * weight[class][i])
module fc_engine #(
    parameter IN_FEATURES    = 784,
    parameter OUT_CLASSES    = 10,
    parameter SRC_ADDR_WIDTH = 10,
    parameter WEIGHT_COUNT   = 7840,
    parameter BIAS_COUNT     = 10,
    parameter WEIGHT_FILE    = "fc_weights.hex",
    parameter BIAS_FILE      = "fc_biases.hex"
) (
    input  wire                      clk,
    input  wire                      rst,
    input  wire                      start,
    output reg                       busy,
    output reg                       done,

    output reg  [SRC_ADDR_WIDTH-1:0] src_addr,
    input  wire [7:0]                src_data,

    output reg                       score_we,
    output reg  [3:0]                score_index,
    output reg signed [31:0]         score_value,

    output wire signed [31:0]        debug_acc
);
    reg signed [7:0]  weights [0:WEIGHT_COUNT-1];
    reg signed [31:0] biases  [0:BIAS_COUNT-1];
    reg signed [7:0]  weight_q;
    reg signed [31:0] accumulator;

    integer class_idx;
    integer feature_idx;
    integer weight_index_calc;

    localparam ST_IDLE  = 3'd0;
    localparam ST_INIT  = 3'd1;
    localparam ST_REQ   = 3'd2;
    localparam ST_MAC   = 3'd3;
    localparam ST_WRITE = 3'd4;
    reg [2:0] state;

    wire signed [8:0] activation_signed = $signed({1'b0, src_data});
    wire signed [16:0] mac_product = activation_signed * weight_q;

    assign debug_acc = accumulator;

    initial begin
        $readmemh(WEIGHT_FILE, weights);
        $readmemh(BIAS_FILE, biases);
    end

    always @* begin
        src_addr = feature_idx;
        weight_index_calc = (class_idx * IN_FEATURES) + feature_idx;
    end

    always @(posedge clk) begin
        if (rst)
            weight_q <= 8'sd0;
        else if (state == ST_REQ)
            weight_q <= weights[weight_index_calc];
    end

    always @(posedge clk) begin
        if (rst) begin
            busy        <= 1'b0;
            done        <= 1'b0;
            score_we    <= 1'b0;
            score_index <= 4'd0;
            score_value <= 32'sd0;
            accumulator <= 32'sd0;
            class_idx   <= 0;
            feature_idx <= 0;
            state       <= ST_IDLE;
        end else begin
            done     <= 1'b0;
            score_we <= 1'b0;

            case (state)
                ST_IDLE: begin
                    if (start) begin
                        busy      <= 1'b1;
                        class_idx <= 0;
                        state     <= ST_INIT;
                    end
                end

                ST_INIT: begin
                    accumulator <= biases[class_idx];
                    feature_idx <= 0;
                    state       <= ST_REQ;
                end

                ST_REQ: begin
                    state <= ST_MAC;
                end

                ST_MAC: begin
                    accumulator <= accumulator + mac_product;
                    if (feature_idx == IN_FEATURES-1) begin
                        state <= ST_WRITE;
                    end else begin
                        feature_idx <= feature_idx + 1;
                        state       <= ST_REQ;
                    end
                end

                ST_WRITE: begin
                    score_we    <= 1'b1;
                    score_index <= class_idx[3:0];
                    score_value <= accumulator;

                    if (class_idx == OUT_CLASSES-1) begin
                        busy  <= 1'b0;
                        done  <= 1'b1;
                        state <= ST_IDLE;
                    end else begin
                        class_idx <= class_idx + 1;
                        state     <= ST_INIT;
                    end
                end

                default: state <= ST_IDLE;
            endcase
        end
    end
endmodule
