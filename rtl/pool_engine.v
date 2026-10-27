`timescale 1ns/1ps

// 2x2 max-pooling, stride 2, channel-major feature-map order.
module pool_engine #(
    parameter IN_W           = 28,
    parameter IN_H           = 28,
    parameter CHANNELS       = 8,
    parameter SRC_ADDR_WIDTH = 13,
    parameter DST_ADDR_WIDTH = 11
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
    output reg  [7:0]                dst_data
);
    localparam OUT_W = IN_W / 2;
    localparam OUT_H = IN_H / 2;

    localparam ST_IDLE    = 3'd0;
    localparam ST_INIT    = 3'd1;
    localparam ST_REQ     = 3'd2;
    localparam ST_CAPTURE = 3'd3;
    localparam ST_WRITE   = 3'd4;
    reg [2:0] state;

    integer channel_idx;
    integer out_row;
    integer out_col;
    integer elem_idx;
    integer source_row_calc;
    integer source_col_calc;
    integer source_index_calc;
    integer dest_index_calc;

    reg [7:0] max_value;
    wire last_output = (channel_idx == CHANNELS-1) &&
                       (out_row == OUT_H-1) && (out_col == OUT_W-1);

    always @* begin
        source_row_calc = (out_row * 2) + ((elem_idx >= 2) ? 1 : 0);
        source_col_calc = (out_col * 2) + (((elem_idx == 1) || (elem_idx == 3)) ? 1 : 0);
        source_index_calc = (channel_idx * IN_H * IN_W) +
                            (source_row_calc * IN_W) + source_col_calc;
        dest_index_calc = (channel_idx * OUT_H * OUT_W) +
                          (out_row * OUT_W) + out_col;
        src_addr = source_index_calc;
    end

    task advance_output;
    begin
        if (out_col == OUT_W-1) begin
            out_col <= 0;
            if (out_row == OUT_H-1) begin
                out_row <= 0;
                if (channel_idx == CHANNELS-1)
                    channel_idx <= 0;
                else
                    channel_idx <= channel_idx + 1;
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
            channel_idx <= 0;
            out_row     <= 0;
            out_col     <= 0;
            elem_idx    <= 0;
            max_value   <= 8'd0;
            state       <= ST_IDLE;
        end else begin
            done   <= 1'b0;
            dst_we <= 1'b0;

            case (state)
                ST_IDLE: begin
                    if (start) begin
                        busy        <= 1'b1;
                        channel_idx <= 0;
                        out_row     <= 0;
                        out_col     <= 0;
                        state       <= ST_INIT;
                    end
                end

                ST_INIT: begin
                    elem_idx  <= 0;
                    max_value <= 8'd0;
                    state     <= ST_REQ;
                end

                ST_REQ: begin
                    state <= ST_CAPTURE;
                end

                ST_CAPTURE: begin
                    if (src_data > max_value)
                        max_value <= src_data;

                    if (elem_idx == 3) begin
                        state <= ST_WRITE;
                    end else begin
                        elem_idx <= elem_idx + 1;
                        state    <= ST_REQ;
                    end
                end

                ST_WRITE: begin
                    dst_we   <= 1'b1;
                    dst_addr <= dest_index_calc;
                    dst_data <= max_value;

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
