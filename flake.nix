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
        simlex-999 = pkgs.fetchzip {
          url = "https://fh295.github.io/SimLex-999.zip";
          sha256 = "sha256-3vCXmkzOm+P/+cy+rYOdUladUBncmej6305Bd4oX3WQ="; # You'll need to update this
          stripRoot = false;
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
              matplotlib
              umap-learn
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
