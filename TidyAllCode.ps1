Get-ChildItem ./src,./include,./apps/src -Recurse -Include *.cpp,*.hpp |
	ForEach-Object {

		Write-Host ''
		Write-Host 'Now processing file' $_.Name...
		Write-Host ''

		$checks = "bugprone-*,"`
		+"modernize-*,-modernize-use-trailing-return-type,-modernize-concat-nested-namespaces,"`
		+"performance-*,"`
		+"readability-*,-readability-magic-numbers,-readability-uppercase-literal-suffix,-readability-avoid-const-params-in-decls,"`
		+"portability-*,-portability-simd-intrinsics,"`
		+"clang-analyzer-*,"`
		+"misc-*,"`
		+"-clang-analyzer-osx*,-clang-analyzer-optin.osx*,-clang-analyzer-optin.mpi*,-clang-analyzer-apiModeling*"
		clang-tidy $_.FullName -p "$PSScriptRoot/build" --checks=$checks -- -march=native -std=c++17 -Iinclude

		Write-Host ''
		Write-Host 'File done, press any key for next file...'
		$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
	}
