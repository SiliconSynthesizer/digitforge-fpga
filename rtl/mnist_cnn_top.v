`timescale 1ns/1ps
`include "quant_params.vh"

module mnist_cnn_top (
    input  wire        clk,
    input  wire        rst,

    input  wire        pixel_valid,
    input  wire [7:0]  pixel_data,
    input  wire [9:0]  pixel_addr,

    input  wire        start,

    output wire        busy,
    output reg         done,
    output reg  [3:0]  predicted_digit,

    output wire signed [31:0] score0,
    output wire signed [31:0] score1,
    output wire signed [31:0] score2,
    output wire signed [31:0] score3,
    output wire signed [31:0] score4,
    output wire signed [31:0] score5,
    output wire signed [31:0] score6,
    output wire signed [31:0] score7,
    output wire signed [31:0] score8,
    output wire signed [31:0] score9,

    output wire [3:0]  debug_state,
    output wire signed [31:0] debug_mac_acc
);
    // -------------------------------
    // Top-level orchestration states
    // -------------------------------
    localparam S_IDLE       = 4'd0;
    localparam S_WAIT_C1    = 4'd1;
    localparam S_WAIT_P1    = 4'd2;
    localparam S_WAIT_C2    = 4'd3;
    localparam S_WAIT_P2    = 4'd4;
    localparam S_WAIT_FC    = 4'd5;
    localparam S_ARGMAX     = 4'd6;
    localparam S_DONE       = 4'd7;

    reg [3:0] state;
    reg top_busy;
    reg [9:0] loaded_pixels;
    reg image_ready;

    assign busy        = top_busy;
    assign debug_state = state;

    // -------------------------------
    // Score registers
    // -------------------------------
    reg signed [31:0] score_r0;
    reg signed [31:0] score_r1;
    reg signed [31:0] score_r2;
    reg signed [31:0] score_r3;
    reg signed [31:0] score_r4;
    reg signed [31:0] score_r5;
    reg signed [31:0] score_r6;
    reg signed [31:0] score_r7;
    reg signed [31:0] score_r8;
    reg signed [31:0] score_r9;

    assign score0 = score_r0;
    assign score1 = score_r1;
    assign score2 = score_r2;
    assign score3 = score_r3;
    assign score4 = score_r4;
    assign score5 = score_r5;
    assign score6 = score_r6;
    assign score7 = score_r7;
    assign score8 = score_r8;
    assign score9 = score_r9;

    // -------------------------------
    // Stage start pulses
    // -------------------------------
    reg conv1_start;
    reg pool1_start;
    reg conv2_start;
    reg pool2_start;
    reg fc_start;

    // -------------------------------
    // Image RAM: 784 x 8
    // -------------------------------
    wire [9:0] conv1_src_addr;
    wire [7:0] image_read_data;
    wire image_we = pixel_valid && !top_busy;

    sync_ram_1r1w #(
        .DATA_WIDTH(8), .ADDR_WIDTH(10), .DEPTH(784)
    ) u_image_mem (
        .clk(clk),
        .we(image_we),
        .waddr(pixel_addr),
        .wdata(pixel_data),
        .raddr(conv1_src_addr),
        .rdata(image_read_data)
    );

    // -------------------------------
    // Conv1 memory: 28*28*8 = 6272
    // -------------------------------
    wire        conv1_dst_we;
    wire [12:0] conv1_dst_addr;
    wire [7:0]  conv1_dst_data;
    wire [12:0] pool1_src_addr;
    wire [7:0]  conv1_read_data;

    sync_ram_1r1w #(
        .DATA_WIDTH(8), .ADDR_WIDTH(13), .DEPTH(6272)
    ) u_conv1_mem (
        .clk(clk),
        .we(conv1_dst_we),
        .waddr(conv1_dst_addr),
        .wdata(conv1_dst_data),
        .raddr(pool1_src_addr),
        .rdata(conv1_read_data)
    );

    // -------------------------------
    // Pool1 memory: 14*14*8 = 1568
    // -------------------------------
    wire        pool1_dst_we;
    wire [10:0] pool1_dst_addr;
    wire [7:0]  pool1_dst_data;
    wire [10:0] conv2_src_addr;
    wire [7:0]  pool1_read_data;

    sync_ram_1r1w #(
        .DATA_WIDTH(8), .ADDR_WIDTH(11), .DEPTH(1568)
    ) u_pool1_mem (
        .clk(clk),
        .we(pool1_dst_we),
        .waddr(pool1_dst_addr),
        .wdata(pool1_dst_data),
        .raddr(conv2_src_addr),
        .rdata(pool1_read_data)
    );

    // -------------------------------
    // Conv2 memory: 14*14*16 = 3136
    // -------------------------------
    wire        conv2_dst_we;
    wire [11:0] conv2_dst_addr;
    wire [7:0]  conv2_dst_data;
    wire [11:0] pool2_src_addr;
    wire [7:0]  conv2_read_data;

    sync_ram_1r1w #(
        .DATA_WIDTH(8), .ADDR_WIDTH(12), .DEPTH(3136)
    ) u_conv2_mem (
        .clk(clk),
        .we(conv2_dst_we),
        .waddr(conv2_dst_addr),
        .wdata(conv2_dst_data),
        .raddr(pool2_src_addr),
        .rdata(conv2_read_data)
    );

    // -------------------------------
    // Pool2 memory: 7*7*16 = 784
    // -------------------------------
    wire        pool2_dst_we;
    wire [9:0]  pool2_dst_addr;
    wire [7:0]  pool2_dst_data;
    wire [9:0]  fc_src_addr;
    wire [7:0]  pool2_read_data;

    sync_ram_1r1w #(
        .DATA_WIDTH(8), .ADDR_WIDTH(10), .DEPTH(784)
    ) u_pool2_mem (
        .clk(clk),
        .we(pool2_dst_we),
        .waddr(pool2_dst_addr),
        .wdata(pool2_dst_data),
        .raddr(fc_src_addr),
        .rdata(pool2_read_data)
    );

    // -------------------------------
    // Compute engines
    // -------------------------------
    wire conv1_busy, conv1_done;
    wire signed [31:0] conv1_debug_acc;
    conv_engine #(
        .IN_W(28), .IN_H(28), .IN_CH(1), .OUT_CH(8),
        .OUT_SHIFT(`CONV1_SHIFT),
        .SRC_ADDR_WIDTH(10), .DST_ADDR_WIDTH(13),
        .WEIGHT_COUNT(72), .BIAS_COUNT(8),
        .WEIGHT_FILE("conv1_weights.hex"),
        .BIAS_FILE("conv1_biases.hex")
    ) u_conv1 (
        .clk(clk), .rst(rst), .start(conv1_start),
        .busy(conv1_busy), .done(conv1_done),
        .src_addr(conv1_src_addr), .src_data(image_read_data),
        .dst_we(conv1_dst_we), .dst_addr(conv1_dst_addr), .dst_data(conv1_dst_data),
        .debug_acc(conv1_debug_acc)
    );

    wire pool1_busy, pool1_done;
    pool_engine #(
        .IN_W(28), .IN_H(28), .CHANNELS(8),
        .SRC_ADDR_WIDTH(13), .DST_ADDR_WIDTH(11)
    ) u_pool1 (
        .clk(clk), .rst(rst), .start(pool1_start),
        .busy(pool1_busy), .done(pool1_done),
        .src_addr(pool1_src_addr), .src_data(conv1_read_data),
        .dst_we(pool1_dst_we), .dst_addr(pool1_dst_addr), .dst_data(pool1_dst_data)
    );

    wire conv2_busy, conv2_done;
    wire signed [31:0] conv2_debug_acc;
    conv_engine #(
        .IN_W(14), .IN_H(14), .IN_CH(8), .OUT_CH(16),
        .OUT_SHIFT(`CONV2_SHIFT),
        .SRC_ADDR_WIDTH(11), .DST_ADDR_WIDTH(12),
        .WEIGHT_COUNT(1152), .BIAS_COUNT(16),
        .WEIGHT_FILE("conv2_weights.hex"),
        .BIAS_FILE("conv2_biases.hex")
    ) u_conv2 (
        .clk(clk), .rst(rst), .start(conv2_start),
        .busy(conv2_busy), .done(conv2_done),
        .src_addr(conv2_src_addr), .src_data(pool1_read_data),
        .dst_we(conv2_dst_we), .dst_addr(conv2_dst_addr), .dst_data(conv2_dst_data),
        .debug_acc(conv2_debug_acc)
    );

    wire pool2_busy, pool2_done;
    pool_engine #(
        .IN_W(14), .IN_H(14), .CHANNELS(16),
        .SRC_ADDR_WIDTH(12), .DST_ADDR_WIDTH(10)
    ) u_pool2 (
        .clk(clk), .rst(rst), .start(pool2_start),
        .busy(pool2_busy), .done(pool2_done),
        .src_addr(pool2_src_addr), .src_data(conv2_read_data),
        .dst_we(pool2_dst_we), .dst_addr(pool2_dst_addr), .dst_data(pool2_dst_data)
    );

    wire fc_busy, fc_done;
    wire fc_score_we;
    wire [3:0] fc_score_index;
    wire signed [31:0] fc_score_value;
    wire signed [31:0] fc_debug_acc;
    fc_engine #(
        .IN_FEATURES(784), .OUT_CLASSES(10),
        .SRC_ADDR_WIDTH(10), .WEIGHT_COUNT(7840), .BIAS_COUNT(10),
        .WEIGHT_FILE("fc_weights.hex"),
        .BIAS_FILE("fc_biases.hex")
    ) u_fc (
        .clk(clk), .rst(rst), .start(fc_start),
        .busy(fc_busy), .done(fc_done),
        .src_addr(fc_src_addr), .src_data(pool2_read_data),
        .score_we(fc_score_we), .score_index(fc_score_index), .score_value(fc_score_value),
        .debug_acc(fc_debug_acc)
    );

    wire [3:0] argmax_digit;
    argmax10 u_argmax (
        .score0(score_r0), .score1(score_r1), .score2(score_r2), .score3(score_r3), .score4(score_r4),
        .score5(score_r5), .score6(score_r6), .score7(score_r7), .score8(score_r8), .score9(score_r9),
        .predicted_digit(argmax_digit)
    );

    assign debug_mac_acc = (state == S_WAIT_C1) ? conv1_debug_acc :
                           (state == S_WAIT_C2) ? conv2_debug_acc :
                           (state == S_WAIT_FC) ? fc_debug_acc    : 32'sd0;

    // Capture FC scores whenever they are produced.
    always @(posedge clk) begin
        if (rst) begin
            score_r0 <= 32'sd0; score_r1 <= 32'sd0; score_r2 <= 32'sd0; score_r3 <= 32'sd0; score_r4 <= 32'sd0;
            score_r5 <= 32'sd0; score_r6 <= 32'sd0; score_r7 <= 32'sd0; score_r8 <= 32'sd0; score_r9 <= 32'sd0;
        end else if (fc_score_we) begin
            case (fc_score_index)
                4'd0: score_r0 <= fc_score_value;
                4'd1: score_r1 <= fc_score_value;
                4'd2: score_r2 <= fc_score_value;
                4'd3: score_r3 <= fc_score_value;
                4'd4: score_r4 <= fc_score_value;
                4'd5: score_r5 <= fc_score_value;
                4'd6: score_r6 <= fc_score_value;
                4'd7: score_r7 <= fc_score_value;
                4'd8: score_r8 <= fc_score_value;
                4'd9: score_r9 <= fc_score_value;
                default: ;
            endcase
        end
    end

    // External loading and stage controller.
    always @(posedge clk) begin
        if (rst) begin
            state           <= S_IDLE;
            top_busy        <= 1'b0;
            done            <= 1'b0;
            predicted_digit <= 4'd0;
            loaded_pixels   <= 10'd0;
            image_ready     <= 1'b0;
            conv1_start     <= 1'b0;
            pool1_start     <= 1'b0;
            conv2_start     <= 1'b0;
            pool2_start     <= 1'b0;
            fc_start        <= 1'b0;
        end else begin
            done        <= 1'b0;
            conv1_start <= 1'b0;
            pool1_start <= 1'b0;
            conv2_start <= 1'b0;
            pool2_start <= 1'b0;
            fc_start    <= 1'b0;

            // Host writes are accepted only while idle. The expected Cocotb
            // protocol sends 784 writes, one per address 0..783.
            if (pixel_valid && !top_busy) begin
                if (loaded_pixels < 10'd784)
                    loaded_pixels <= loaded_pixels + 1'b1;
                if (loaded_pixels == 10'd783)
                    image_ready <= 1'b1;
            end

            case (state)
                S_IDLE: begin
                    top_busy <= 1'b0;
                    if (start && image_ready && !pixel_valid) begin
                        top_busy      <= 1'b1;
                        loaded_pixels <= 10'd0;
                        image_ready   <= 1'b0;
                        conv1_start   <= 1'b1;
                        state         <= S_WAIT_C1;
                    end
                end

                S_WAIT_C1: begin
                    top_busy <= 1'b1;
                    if (conv1_done) begin
                        pool1_start <= 1'b1;
                        state       <= S_WAIT_P1;
                    end
                end

                S_WAIT_P1: begin
                    top_busy <= 1'b1;
                    if (pool1_done) begin
                        conv2_start <= 1'b1;
                        state       <= S_WAIT_C2;
                    end
                end

                S_WAIT_C2: begin
                    top_busy <= 1'b1;
                    if (conv2_done) begin
                        pool2_start <= 1'b1;
                        state       <= S_WAIT_P2;
                    end
                end

                S_WAIT_P2: begin
                    top_busy <= 1'b1;
                    if (pool2_done) begin
                        fc_start <= 1'b1;
                        state    <= S_WAIT_FC;
                    end
                end

                S_WAIT_FC: begin
                    top_busy <= 1'b1;
                    if (fc_done)
                        state <= S_ARGMAX;
                end

                S_ARGMAX: begin
                    top_busy        <= 1'b1;
                    predicted_digit <= argmax_digit;
                    state           <= S_DONE;
                end

                S_DONE: begin
                    top_busy <= 1'b0;
                    done     <= 1'b1;
                    state    <= S_IDLE;
                end

                default: begin
                    top_busy <= 1'b0;
                    state    <= S_IDLE;
                end
            endcase
        end
    end
endmodule
