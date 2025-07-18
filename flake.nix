{
  description = "Implementation of the word2vec algorithm";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    rust-overlay,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        overlays = [(import rust-overlay)];
        # legPkgs = nixpkgs.legacyPackages.${system};

        pkgs = import nixpkgs {
          inherit system overlays;
        };
        rust-bin-custom = pkgs.rust-bin.stable.latest.default.override {
          extensions = ["rust-src"];
        };
      in {
        devShells.default = pkgs.mkShell {
          venvDir = ".venv";

          packages = with pkgs; [
            (with python312Packages; [
              venvShellHook
              numpy
              pandas
              plotly
              scikit-learn
              networkx
            ])
          ];

          buildInputs = [
            pkgs.cargo-show-asm
            pkgs.openssl
            pkgs.pkg-config
            rust-bin-custom
          ];
        };
      }
    );
}
