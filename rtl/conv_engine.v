`timescale 1ns/1ps

// Generic 3x3, stride-1, padding-1 convolution engine.
// - UINT8 input activation
// - INT8 weight
// - INT32 bias/accumulator
// - ReLU + arithmetic right shift + UINT8 saturation
// - One MAC result is consumed every two clocks (request/read then MAC)
module conv_engine #(
    parameter IN_W           = 28,
    parameter IN_H           = 28,
    parameter IN_CH          = 1,
    parameter OUT_CH         = 8,
    parameter OUT_SHIFT      = 8,
    parameter SRC_ADDR_WIDTH = 10,
    parameter DST_ADDR_WIDTH = 13,
    parameter WEIGHT_COUNT   = 72,
    parameter BIAS_COUNT     = 8,
    parameter WEIGHT_FILE    = "conv1_weights.hex",
    parameter BIAS_FILE      = "conv1_biases.hex"
) (
    input  wire                      clk,
    input  wire                      rst,
    input  wire                      start,
    output reg                       busy,
    output reg                       done,

    output reg  [SRC_ADDR_WIDTH-1:0] src_addr,
    input  wire [7:0]                src_data,

    output reg                       dst_we,
    output reg  [DST_ADDR_WIDTH-1:0] dst_addr,
    output reg  [7:0]                dst_data,

    output wire signed [31:0]        debug_acc
);
    reg signed [7:0]  weights [0:WEIGHT_COUNT-1];
    reg signed [31:0] biases  [0:BIAS_COUNT-1];
    reg signed [7:0]  weight_q;
    reg signed [31:0] accumulator;

    integer out_ch_idx;
    integer out_row;
    integer out_col;
    integer in_ch_idx;
    integer kernel_row;
    integer kernel_col;

    integer input_row_calc;
    integer input_col_calc;
    integer source_index_calc;
    integer weight_index_calc;
    integer dest_index_calc;
    reg     tap_valid;

    localparam ST_IDLE  = 3'd0;
    localparam ST_INIT  = 3'd1;
    localparam ST_REQ   = 3'd2;
    localparam ST_MAC   = 3'd3;
    localparam ST_WRITE = 3'd4;
    reg [2:0] state;

    wire signed [8:0] activation_signed = $signed({1'b0, src_data});
    wire signed [16:0] mac_product = activation_signed * weight_q;
    wire signed [31:0] shifted_acc = accumulator >>> OUT_SHIFT;
    wire last_tap = (in_ch_idx == IN_CH-1) &&
                    (kernel_row == 2) && (kernel_col == 2);
    wire last_output = (out_ch_idx == OUT_CH-1) &&
                       (out_row == IN_H-1) && (out_col == IN_W-1);

    assign debug_acc = accumulator;

    initial begin
        $readmemh(WEIGHT_FILE, weights);
        $readmemh(BIAS_FILE, biases);
    end

    always @* begin
        input_row_calc  = out_row + kernel_row - 1;
        input_col_calc  = out_col + kernel_col - 1;
        tap_valid       = 1'b0;
        source_index_calc = 0;
        weight_index_calc = (((out_ch_idx * IN_CH) + in_ch_idx) * 3 + kernel_row) * 3 + kernel_col;
        dest_index_calc = (out_ch_idx * IN_H * IN_W) + (out_row * IN_W) + out_col;

        if ((input_row_calc >= 0) && (input_row_calc < IN_H) &&
            (input_col_calc >= 0) && (input_col_calc < IN_W)) begin
            tap_valid = 1'b1;
            source_index_calc = (in_ch_idx * IN_H * IN_W) +
                                (input_row_calc * IN_W) + input_col_calc;
        end

        src_addr = source_index_calc;
    end

    // Synchronous weight-ROM read, aligned with the source RAM's one-cycle read.
    always @(posedge clk) begin
        if (rst)
            weight_q <= 8'sd0;
        else if ((state == ST_REQ) && tap_valid)
            weight_q <= weights[weight_index_calc];
    end

    task advance_tap;
    begin
        if (kernel_col == 2) begin
            kernel_col <= 0;
            if (kernel_row == 2) begin
                kernel_row <= 0;
                if (in_ch_idx == IN_CH-1)
                    in_ch_idx <= 0;
                else
                    in_ch_idx <= in_ch_idx + 1;
            end else begin
                kernel_row <= kernel_row + 1;
            end
        end else begin
            kernel_col <= kernel_col + 1;
        end
    end
    endtask

    task advance_output;
    begin
        if (out_col == IN_W-1) begin
            out_col <= 0;
            if (out_row == IN_H-1) begin
                out_row <= 0;
                if (out_ch_idx == OUT_CH-1)
                    out_ch_idx <= 0;
                else
                    out_ch_idx <= out_ch_idx + 1;
            end else begin
                out_row <= out_row + 1;
            end
        end else begin
            out_col <= out_col + 1;
        end
    end
    endtask

    always @(posedge clk) begin
        if (rst) begin
            busy        <= 1'b0;
            done        <= 1'b0;
            dst_we      <= 1'b0;
            dst_addr    <= {DST_ADDR_WIDTH{1'b0}};
            dst_data    <= 8'd0;
            accumulator <= 32'sd0;
            out_ch_idx  <= 0;
            out_row     <= 0;
            out_col     <= 0;
            in_ch_idx   <= 0;
            kernel_row  <= 0;
            kernel_col  <= 0;
            state       <= ST_IDLE;
        end else begin
            done   <= 1'b0;
            dst_we <= 1'b0;

            case (state)
                ST_IDLE: begin
                    if (start) begin
                        busy       <= 1'b1;
                        out_ch_idx <= 0;
                        out_row    <= 0;
                        out_col    <= 0;
                        state      <= ST_INIT;
                    end
                end

                ST_INIT: begin
                    accumulator <= biases[out_ch_idx];
                    in_ch_idx   <= 0;
                    kernel_row  <= 0;
                    kernel_col  <= 0;
                    state       <= ST_REQ;
                end

                ST_REQ: begin
                    if (tap_valid) begin
                        state <= ST_MAC;
                    end else if (last_tap) begin
                        state <= ST_WRITE;
                    end else begin
                        advance_tap;
                    end
                end

                ST_MAC: begin
                    accumulator <= accumulator + mac_product;
                    if (last_tap) begin
                        state <= ST_WRITE;
                    end else begin
                        advance_tap;
                        state <= ST_REQ;
                    end
                end

                ST_WRITE: begin
                    dst_we   <= 1'b1;
                    dst_addr <= dest_index_calc;

                    if (accumulator <= 0)
                        dst_data <= 8'd0;
                    else if (shifted_acc > 32'sd255)
                        dst_data <= 8'hFF;
                    else
                        dst_data <= shifted_acc[7:0];

                    if (last_output) begin
                        busy  <= 1'b0;
                        done  <= 1'b1;
                        state <= ST_IDLE;
                    end else begin
                        advance_output;
                        state <= ST_INIT;
                    end
                end

                default: state <= ST_IDLE;
            endcase
        end
    end
endmodule
