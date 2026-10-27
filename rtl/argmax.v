`timescale 1ns/1ps

// Combinational argmax over ten signed logits. Ties keep the lower index.
module argmax10 (
    input  wire signed [31:0] score0,
    input  wire signed [31:0] score1,
    input  wire signed [31:0] score2,
    input  wire signed [31:0] score3,
    input  wire signed [31:0] score4,
    input  wire signed [31:0] score5,
    input  wire signed [31:0] score6,
    input  wire signed [31:0] score7,
    input  wire signed [31:0] score8,
    input  wire signed [31:0] score9,
    output reg  [3:0]         predicted_digit
);
    reg signed [31:0] max_score;

    always @* begin
        max_score       = score0;
        predicted_digit = 4'd0;
        if (score1 > max_score) begin max_score = score1; predicted_digit = 4'd1; end
        if (score2 > max_score) begin max_score = score2; predicted_digit = 4'd2; end
        if (score3 > max_score) begin max_score = score3; predicted_digit = 4'd3; end
        if (score4 > max_score) begin max_score = score4; predicted_digit = 4'd4; end
        if (score5 > max_score) begin max_score = score5; predicted_digit = 4'd5; end
        if (score6 > max_score) begin max_score = score6; predicted_digit = 4'd6; end
        if (score7 > max_score) begin max_score = score7; predicted_digit = 4'd7; end
        if (score8 > max_score) begin max_score = score8; predicted_digit = 4'd8; end
        if (score9 > max_score) begin max_score = score9; predicted_digit = 4'd9; end
    end
endmodule
