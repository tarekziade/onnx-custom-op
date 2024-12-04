import onnx
from onnx import helper


new_name = "FirefoxMatMulInteger8"


def replace_matmul_with_firefoxmatmul(onnx_file_path, output_file_path):
    # Load the ONNX model
    model = onnx.load(onnx_file_path)
    graph = model.graph

    new_nodes = []
    for node in graph.node:
        if node.op_type == "MatMulInteger":
            op_type = node.op_type
            # Replace MatMul* with FirefoxMatMul*
            firefox_matmul_node = helper.make_node(
                new_name,
                domain="com.microsoft",
                inputs=node.input,
                outputs=node.output,
                name=node.name if node.name else None,
            )
            new_nodes.append(firefox_matmul_node)
            print(f"Replaced {op_type} with {new_name}")
            print(firefox_matmul_node)
        else:
            # Keep other nodes unchanged
            new_nodes.append(node)

    # Replace the graph's nodes with the updated nodes
    graph.ClearField("node")
    graph.node.extend(new_nodes)

    # Save the modified ONNX model
    onnx.save(model, output_file_path)
    print(f"Updated model saved to {output_file_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Replace MatMul with FirefoxMatMul in an ONNX model."
    )
    parser.add_argument("input", help="Path to the input ONNX model file.")
    parser.add_argument("output", help="Path to save the updated ONNX model file.")
    args = parser.parse_args()

    replace_matmul_with_firefoxmatmul(args.input, args.output)
